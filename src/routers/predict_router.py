from fastapi import APIRouter
from schemas.patient_schema import PatientSchema
from services.predict_service import predict_result

router = APIRouter()

@router.post("/")
def predict(patient: PatientSchema):
    result = predict_result(patient.model_dump())
    return {"prediction": result}
