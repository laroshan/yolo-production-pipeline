from datetime import datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.active_learning import ActiveLearningRecord
from app.schemas.active_learning import (
    ActiveLearningFeedbackRequest,
    ActiveLearningQueueItem,
    ReviewStatus,
)
from app.schemas.inference import InferenceResponse

logger = structlog.get_logger(__name__)


class ActiveLearningService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def ingest_inference_result(self, response: InferenceResponse) -> None:
        """
        Evaluates detections. If any prediction matches uncertainty thresholds,
        records it in the review queue for reviewer correction.
        """
        uncertain_detections = [d for d in response.detections if d.is_uncertain]

        if not uncertain_detections:
            return

        for item in uncertain_detections:
            record = ActiveLearningRecord(
                image_id=response.image_id,
                filename=response.filename,
                detected_class=item.class_name,
                confidence=item.confidence,
                status=ReviewStatus.PENDING_REVIEW.value,
            )
            record.set_raw_predictions([d.model_dump() for d in response.detections])
            self.db.add(record)

        await self.db.commit()
        logger.info(
            "active_learning_flagged_for_triage",
            image_id=response.image_id,
            count=len(uncertain_detections),
        )

    async def get_pending_queue(self, limit: int = 50) -> list[ActiveLearningQueueItem]:
        """Fetches pending items for the React frontend 'Violation ML Solution Page'."""
        query = (
            select(ActiveLearningRecord)
            .where(ActiveLearningRecord.status == ReviewStatus.PENDING_REVIEW.value)
            .order_by(ActiveLearningRecord.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(query)
        records = result.scalars().all()

        return [
            ActiveLearningQueueItem(
                id=r.id,
                image_id=r.image_id,
                filename=r.filename,
                confidence=r.confidence,
                detected_class=r.detected_class,
                status=ReviewStatus(r.status),
                created_at=r.created_at,
                reviewed_by=r.reviewer_id,
                reviewed_at=r.reviewed_at,
            )
            for r in records
        ]

    async def submit_feedback(self, feedback: ActiveLearningFeedbackRequest) -> bool:
        """
        Updates database record when an analyst adjusts bounding boxes in React.
        Marks records as VALIDATED and stores adjusted coordinates for the next SageMaker training run.
        """
        query = select(ActiveLearningRecord).where(
            ActiveLearningRecord.image_id == feedback.image_id
        )
        result = await self.db.execute(query)
        records = result.scalars().all()

        if not records:
            # If image was not originally flagged as uncertain, create a ground-truth entry
            new_record = ActiveLearningRecord(
                image_id=feedback.image_id,
                filename=f"{feedback.image_id}.jpg",
                detected_class=feedback.corrected_boxes[0].class_name if feedback.corrected_boxes else "manual_label",
                confidence=1.0,
                status=ReviewStatus.VALIDATED.value,
                reviewer_id=feedback.reviewer_id,
                claim_reference_id=feedback.claim_reference_id,
                review_notes=feedback.review_notes,
                reviewed_at=datetime.utcnow(),
            )
            new_record.set_corrected_boxes([b.model_dump() for b in feedback.corrected_boxes])
            self.db.add(new_record)
        else:
            for r in records:
                r.status = ReviewStatus.VALIDATED.value
                r.reviewer_id = feedback.reviewer_id
                r.claim_reference_id = feedback.claim_reference_id
                r.review_notes = feedback.review_notes
                r.reviewed_at = datetime.utcnow()
                r.set_corrected_boxes([b.model_dump() for b in feedback.corrected_boxes])

        await self.db.commit()
        logger.info(
            "active_learning_feedback_recorded",
            image_id=feedback.image_id,
            reviewer=feedback.reviewer_id,
        )
        return True
