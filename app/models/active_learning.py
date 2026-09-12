import json
from datetime import datetime
import uuid
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from app.db.session import Base


class ActiveLearningRecord(Base):
    __tablename__ = "active_learning_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(64), index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    detected_class = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(32), default="PENDING_REVIEW", index=True)
    
    # Raw prediction serialized bounding boxes
    raw_prediction_json = Column(Text, nullable=True)
    
    # Human reviewer feedback
    reviewer_id = Column(String(64), nullable=True, index=True)
    claim_reference_id = Column(String(64), nullable=True)
    corrected_boxes_json = Column(Text, nullable=True)
    review_notes = Column(Text, nullable=True)
    
    # Audit timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)

    def set_raw_predictions(self, data: list):
        self.raw_prediction_json = json.dumps(data)

    def get_raw_predictions(self) -> list:
        return json.loads(self.raw_prediction_json) if self.raw_prediction_json else []

    def set_corrected_boxes(self, data: list):
        self.corrected_boxes_json = json.dumps(data)

    def get_corrected_boxes(self) -> list:
        return json.loads(self.corrected_boxes_json) if self.corrected_boxes_json else []
