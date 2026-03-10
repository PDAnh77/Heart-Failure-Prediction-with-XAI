from fastapi import APIRouter, Depends, Query
from services.auth_service import require_roles
from schemas.patient_schema import PatientCreate, PatientUpdate
from services.patient_service import (
    get_patients_service,
    get_patient_service,
    get_patients_by_user,
    create_patient_service,
    update_patient_service,
    delete_patient_service,
    delete_user_patients_service,
    get_random_patient_service
)

router = APIRouter()

@router.get("/", dependencies=[Depends(require_roles(["admin"]))])
def get_patients(
    limit: int = Query(10, ge=1, le=100, description="Number of patient records per page"),
    offset: int = Query(0, ge=0, description="Starting index")):
    return get_patients_service(limit, offset)

@router.get("/rand", dependencies=[Depends(require_roles(["admin", "viewer"]))])
def get_random_patient():
    return get_random_patient_service()

@router.get("/me")
def get_user_patients_me(user = Depends(require_roles(["admin", "viewer"]))):
    return get_patients_by_user(user["user_id"])

@router.get("/users/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def get_user_patients_admin(user_id: str):
    return get_patients_by_user(user_id)

@router.get("/{patient_id}")
def get_patient(patient_id: str, user = Depends(require_roles(["admin"]))):
    return get_patient_service(patient_id, user)

@router.post("/")
def create_patient(new_patient: PatientCreate, user = Depends(require_roles(["admin", "viewer"]))):
    return create_patient_service(new_patient, user["user_id"])

@router.put("/{patient_id}")
def update_patient(patient_id: str, patient: PatientUpdate, user = Depends(require_roles(["admin", "viewer"]))):
    return update_patient_service(patient_id, patient, user)

@router.delete("/{patient_id}")
def delete_patient(patient_id: str, user = Depends(require_roles(["admin", "viewer"]))):
    return delete_patient_service(patient_id, user)

@router.delete("/users/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def delete_user_patients(user_id: str):
    return delete_user_patients_service(user_id)
