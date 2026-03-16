from sqlalchemy import Column, Float, ForeignKey, Integer, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
from db.database import Base
import uuid


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    input_data = Column(JSONB, nullable=False)
    prediction_xai = Column(JSONB, nullable=False)
    predicted_label = Column(Integer, nullable=True)
    predicted_probability = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
