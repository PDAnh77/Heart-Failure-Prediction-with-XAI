from sqlalchemy import Column, Float, ForeignKey, Integer, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from db.database import Base
import uuid


class Patient(Base):
    __tablename__ = "patients"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    age = Column(Integer, nullable=True)
    sex = Column(String, nullable=True)
    chest_pain_type = Column(String, nullable=True)
    resting_bp = Column(Integer, nullable=True)
    cholesterol = Column(Integer, nullable=True)
    fasting_bs = Column(Integer, nullable=True)
    resting_ecg = Column(String, nullable=True)
    max_hr = Column(Integer, nullable=True)
    exercise_angina = Column(String, nullable=True)
    oldpeak = Column(Float, nullable=True)
    st_slope = Column(String, nullable=True)
    heart_disease = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
