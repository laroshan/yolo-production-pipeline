from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBoxCoordinates(BaseModel):
    """Normalized coordinates [0.0 - 1.0] for responsive frontend rendering."""
    x_min: float = Field(..., description="Normalized top-left X")
    y_min: float = Field(..., description="Normalized top-left Y")
    x_max: float = Field(..., description="Normalized bottom-right X")
    y_max: float = Field(..., description="Normalized bottom-right Y")


class PixelCoordinates(BaseModel):
    """Raw pixel boundaries."""
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    normalized_box: BoundingBoxCoordinates
    pixel_box: PixelCoordinates
    is_uncertain: bool = Field(
        False,
        description="True if prediction confidence lands within active learning uncertainty band."
    )


class InferenceResponse(BaseModel):
    image_id: str
    filename: str
    image_width: int
    image_height: int
    total_detections: int
    detections: List[DetectionItem]
    requires_human_triage: bool
    inference_time_ms: float
    model_version: str
