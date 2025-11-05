from typing import List
from fastapi import APIRouter
from schemas.patient_schema import PatientSchema
from services.predict_service import predict_result

router = APIRouter()

@router.post("/")
def predict(patient: PatientSchema):
    result = predict_result(patient.model_dump())
    return {"prediction": result}

@router.post("/batch")
def predict_batch(patients: List[PatientSchema]):
    patient_data_list = [p.model_dump() for p in patients]
    predictions_list = predict_result(patient_data_list)
    output = [{"prediction": int(pred)} for pred in predictions_list]
    return output
