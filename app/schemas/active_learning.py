from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from app.schemas.inference import BoundingBoxCoordinates


class ReviewStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    EXPORTED_TO_S3 = "EXPORTED_TO_S3"


class CorrectedBox(BaseModel):
    class_name: str
    normalized_box: BoundingBoxCoordinates


class ActiveLearningFeedbackRequest(BaseModel):
    """Payload received directly from React frontend 'Violation ML Solution Page'."""
    image_id: str
    reviewer_id: str
    claim_reference_id: str | None = None
    corrected_boxes: list[CorrectedBox]
    review_notes: str | None = None


class ActiveLearningQueueItem(BaseModel):
    id: str
    image_id: str
    filename: str
    confidence: float
    detected_class: str
    status: ReviewStatus
    created_at: datetime
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None


class ActiveLearningQueueResponse(BaseModel):
    total_pending: int
    items: list[ActiveLearningQueueItem]
