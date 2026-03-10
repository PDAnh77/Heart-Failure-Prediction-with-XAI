from fastapi import APIRouter, Depends, Query
from services.auth_service import require_roles
from services.prediction_history_service import(
    get_user_predictions_service,
    get_prediction_service,
    delete_prediction_service,
    delete_user_predictions_service
)

router = APIRouter()

@router.get("/me")
def get_user_predictions_me(
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
    user: str = Depends(require_roles(["admin", "viewer"]))):
    return get_user_predictions_service(user["user_id"], limit, offset)

@router.get("/users/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def get_user_predictions_admin(
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0)):
    return get_user_predictions_service(user_id, limit, offset)

@router.get("/{prediction_id}")
def get_prediction(prediction_id: str, user = Depends(require_roles(["admin", "viewer"]))):
    return get_prediction_service(prediction_id, user)

@router.delete("/me")
def delete_user_predictions_me(user = Depends(require_roles(["admin", "viewer"]))):
    return delete_user_predictions_service(user["user_id"])

@router.delete("/users/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def delete_user_predictions_admin(user_id: str):
    return delete_user_predictions_service(user_id)

@router.delete("/{prediction_id}")
def delete_prediction(prediction_id: str, user = Depends(require_roles(["admin", "viewer"]))):
    return delete_prediction_service(prediction_id, user)