from datetime import datetime
from typing import Any, Dict 
from pydantic import BaseModel

class PredictionBase(BaseModel):
    id: str
    user_id: str
    created_at: datetime

class PredictionGet(BaseModel):
    input_data: Dict[str, Any]
    predicted_label: int
    predicted_probability: float
    prediction_xai: Dict[str, Any]