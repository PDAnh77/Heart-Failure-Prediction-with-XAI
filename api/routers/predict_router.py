from typing import List
from fastapi import APIRouter, Depends
from services.auth_service import validate_token
from schemas.patient_schema import PatientBase, PatientPredict
from services.predict_service import predict_result

router = APIRouter()

@router.post("")
def create_prediction(patient: PatientPredict, user_id: str = Depends(validate_token)):
    result = predict_result(patient.model_dump(), user_id)
    return result

@router.post("/batch")
def create_batch_prediction(patients: List[PatientBase]):
    patient_data_list = [patient.model_dump() for patient in patients]
    return predict_result(patient_data_list)
