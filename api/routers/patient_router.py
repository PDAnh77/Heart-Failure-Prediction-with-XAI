from typing import List
from fastapi import APIRouter, Depends, Query
from services.auth_service import require_roles
from schemas.patient_schema import PatientCreate, PatientGet, PatientUpdate
from dependencies import get_db
from sqlalchemy.orm import Session
from services import patient_service

router = APIRouter()


@router.get("/", dependencies=[Depends(require_roles(["admin"]))], response_model=List[PatientGet])
def list_patients(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return patient_service.get_patients(db, limit, offset)


@router.get("/rand", dependencies=[Depends(require_roles(["admin", "user"]))], response_model=PatientGet)
def get_random_patient(db: Session = Depends(get_db)):
    return patient_service.get_random_patient(db)


@router.get("/me", response_model=List[PatientGet])
def get_user_patients_me(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user=Depends(require_roles(["admin", "user"])),
    db: Session = Depends(get_db),
):
    return patient_service.get_patients_by_user(db, user["user_id"], limit, offset)


@router.get("/users/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def get_user_patients_admin(
    user_id: str,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)):
    return patient_service.get_patients_by_user(db, user_id, limit, offset)


@router.get("/{patient_id}", response_model=PatientGet)
def get_patient(patient_id: str, user=Depends(require_roles(["admin"])), db: Session = Depends(get_db)):
    return patient_service.get_patient_by_id(db, patient_id, user)


@router.post("/")
def create_patient(
    new_patient: PatientCreate,
    user=Depends(require_roles(["admin", "user"])),
    db: Session = Depends(get_db),
):
    return patient_service.create_patient(db, new_patient, user["user_id"])


@router.patch("/{patient_id}")
def update_patient(
    patient_id: str,
    patient: PatientUpdate,
    user=Depends(require_roles(["admin", "user"])),
    db: Session = Depends(get_db),
):
    return patient_service.update_patient(db, patient_id, patient, user)


@router.delete("/{patient_id}")
def delete_patient(
    patient_id: str,
    user=Depends(require_roles(["admin", "user"])),
    db: Session = Depends(get_db),
):
    return patient_service.delete_patient(db, patient_id, user)


@router.delete("/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def delete_user_patients(user_id: str, db: Session = Depends(get_db)):
    return patient_service.delete_patients_by_user(db, user_id)
