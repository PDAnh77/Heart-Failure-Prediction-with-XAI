from fastapi import APIRouter, Depends, Query
from services.auth_service import validate_token
from services.prediction_history_service import(
    get_user_predictions_service,
    get_info_prediction_service,
    delete_prediction_service,
    delete_user_predictions_service
)

router = APIRouter()

@router.get("")
def get_user_predictions(
    limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(validate_token)):
    return get_user_predictions_service(user_id, limit, offset)

@router.get("/{prediction_id}")
def get_prediction(prediction_id: str):
    return get_info_prediction_service(prediction_id)

@router.delete("")
def delete_user_predictions(user_id: str = Depends(validate_token)):
    return delete_user_predictions_service(user_id)

@router.delete("/{prediction_id}")
def delete_prediction(prediction_id: str):
    return delete_prediction_service(prediction_id)