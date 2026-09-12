from typing import List
from fastapi import APIRouter, Depends, Query, status
import structlog

from app.api.deps import get_active_learning_service
from app.schemas.active_learning import (
    ActiveLearningFeedbackRequest,
    ActiveLearningQueueItem,
    ActiveLearningQueueResponse,
)
from app.schemas.common import APIResponse
from app.services.active_learning_service import ActiveLearningService

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get(
    "/queue",
    response_model=ActiveLearningQueueResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve low-confidence queue for human triage",
    description="Returns list of uncertain predictions awaiting manual bounding-box review on React frontend.",
)
async def get_review_queue(
    limit: int = Query(50, ge=1, le=200),
    service: ActiveLearningService = Depends(get_active_learning_service),
):
    items = await service.get_pending_queue(limit=limit)
    return ActiveLearningQueueResponse(total_pending=len(items), items=items)


@router.post(
    "/correction",
    response_model=APIResponse[dict],
    status_code=status.HTTP_200_OK,
    summary="Submit ground-truth bounding box corrections",
    description="Direct microservice integration: React frontend submits corrected annotations for next SageMaker training cycle.",
)
async def submit_annotation_correction(
    feedback: ActiveLearningFeedbackRequest,
    service: ActiveLearningService = Depends(get_active_learning_service),
):
    await service.submit_feedback(feedback)
    return APIResponse(
        success=True,
        message="Correction stored successfully in ground-truth registry.",
        data={"image_id": feedback.image_id, "reviewer_id": feedback.reviewer_id},
    )
