from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
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
    claim_reference_id: Optional[str] = None
    corrected_boxes: List[CorrectedBox]
    review_notes: Optional[str] = None


class ActiveLearningQueueItem(BaseModel):
    id: str
    image_id: str
    filename: str
    confidence: float
    detected_class: str
    status: ReviewStatus
    created_at: datetime
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class ActiveLearningQueueResponse(BaseModel):
    total_pending: int
    items: List[ActiveLearningQueueItem]
