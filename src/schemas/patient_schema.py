from pydantic import BaseModel, Field
from typing import Optional

class PatientSchema(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: str = Field(..., pattern="^(M|F)")
    chest_pain_type: str = Field(...)
    resting_bp: int = Field(..., gt=0)
    cholesterol: int = Field(..., gt=0)
    fasting_bs: int = Field(..., ge=0, le=1)
    resting_ecg: str = Field(...)
    max_hr: int = Field(..., gt=0)
    exercise_angina: str = Field(...)
    oldpeak: float = Field(..., ge=0.0)
    st_slope: str = Field(...)

class PatientGet(PatientSchema):
    heart_disease: Optional[int] = Field(None, ge=0, le=1)
    
class PatientCreate(PatientSchema):
    heart_disease: Optional[int] = Field(None, ge=0, le=1)