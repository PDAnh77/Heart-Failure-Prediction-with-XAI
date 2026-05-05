from typing import List
from fastapi import APIRouter, Depends, Query, UploadFile, File
from schemas.prediction_schema import PredictionBase, PredictionGet
from services.auth_service import require_roles
from dependencies import get_db
from sqlalchemy.orm import Session
from schemas.patient_schema import PatientBase, PatientPredict
from services.prediction_service import (
    get_predictions_by_user,
    get_prediction_by_id,
    delete_prediction_by_id,
    delete_predictions_by_user,
    predict_dataframe,
    predict_single,
    predict_batch,
)

router = APIRouter()


@router.get("/me", response_model=List[PredictionBase])
def get_user_predictions_me(
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
    user: str = Depends(require_roles(["admin", "user"])),
    db: Session = Depends(get_db),
):
    return get_predictions_by_user(db, user["user_id"], limit, offset)


@router.get("/users/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def get_user_predictions_admin(
    user_id: str, limit: int = Query(10, ge=1, le=50), offset: int = Query(0, ge=0), db: Session = Depends(get_db)
):
    return get_predictions_by_user(db, user_id, limit, offset)


@router.get("/{prediction_id}", response_model=PredictionGet)
def get_prediction(prediction_id: str, user=Depends(require_roles(["admin", "user"])), db: Session = Depends(get_db)):
    return get_prediction_by_id(db, prediction_id, user)


@router.post("")
def create_prediction(
    patient: PatientPredict, user=Depends(require_roles(["admin", "user"])), db: Session = Depends(get_db)
):
    return predict_single(db, patient, user["user_id"])


@router.post("/batch", dependencies=[Depends(require_roles(["admin"]))])
def create_batch_prediction(patients: List[PatientBase]):
    return predict_batch(patients)


@router.post("/upload")
def create_prediction_from_file(
    dataset_id: str, target_column: str = Query(None), user=Depends(require_roles(["admin", "user"]))
):
    return predict_dataframe(dataset_id, user["user_id"], target_column)


@router.delete("/me")
def delete_user_predictions_me(user=Depends(require_roles(["admin", "user"])), db: Session = Depends(get_db)):
    return delete_predictions_by_user(db, user["user_id"])


@router.delete("/users/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def delete_user_predictions_admin(user_id: str, db: Session = Depends(get_db)):
    return delete_predictions_by_user(db, user_id)


@router.delete("/{prediction_id}")
def delete_prediction(
    prediction_id: str, user=Depends(require_roles(["admin", "user"])), db: Session = Depends(get_db)
):
    return delete_prediction_by_id(db, prediction_id, user)
