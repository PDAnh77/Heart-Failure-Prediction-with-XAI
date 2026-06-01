from datetime import datetime
from typing import Any, Dict
from uuid import UUID
from pydantic import BaseModel


class PredictionBase(BaseModel):
    id: UUID
    user_id: UUID
    created_at: datetime


class PredictionGet(BaseModel):
    input_data: Dict[str, Any]
    predicted_label: int
    predicted_probability: float
    prediction_xai: Dict[str, Any]
    created_at: datetime


class UnifiedHistoryItem(BaseModel):
    id: UUID
    type: str
    created_at: datetime
