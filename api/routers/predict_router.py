from typing import List
from fastapi import APIRouter, Depends
from services.auth_service import require_roles
from schemas.patient_schema import PatientBase, PatientPredict
from services.predict_service import predict_single, predict_batch

router = APIRouter()

@router.post("")
def create_prediction(patient: PatientPredict, user = Depends(require_roles(["admin", "viewer"]))):
    return predict_single(patient, user["user_id"])

@router.post("/batch", dependencies=[Depends(require_roles(["admin"]))])
def create_batch_prediction(patients: List[PatientBase]):
    return predict_batch(patients)
