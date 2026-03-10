from uuid import UUID
from fastapi import HTTPException, status
from schemas.user_schema import UserBase
from db.database import supabase
from schemas.prediction_schema import PredictionBase, PredictionGet

TABLE_PREDICTION = "prediction_histories"
TABLE_USER = "users"

def check_uuid(id: str):
    try:
        validated_uuid = str(UUID(id))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format")
    return validated_uuid

def get_user_predictions_service(user_id: str, limit: int, offset: int):
    user_uuid = check_uuid(user_id)
    columns = ','.join(PredictionBase.model_fields.keys())
    result = supabase.table(TABLE_PREDICTION).select(columns).eq("user_id", user_uuid).range(offset, offset + limit - 1).order("created_at", desc=True).execute()
    return result.data

def get_prediction_service(prediction_id: str, user: dict):
    prediction_uuid = check_uuid(prediction_id)
    columns = ','.join(PredictionGet.model_fields.keys())

    if user["role"] == "admin":
        result = supabase.table(TABLE_PREDICTION).select(columns).eq("id", prediction_uuid).execute()
    else:
        result = supabase.table(TABLE_PREDICTION).select(columns).eq("id", prediction_uuid).eq("user_id", user["user_id"]).execute()
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return result.data[0]

def delete_prediction_service(prediction_id: str, user: dict):
    prediction_uuid = check_uuid(prediction_id)
    if user["role"] == "admin":
        result = supabase.table(TABLE_PREDICTION).delete().eq("id", prediction_uuid).execute()
    else:
        result = supabase.table(TABLE_PREDICTION).delete().eq("id", prediction_uuid).eq("user_id", user["user_id"]).execute()
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return {"detail": "Delete user prediction history successfully"}

def delete_user_predictions_service(user_id: str):
    user_uuid = check_uuid(user_id)

    columns = ",".join(UserBase.model_fields.keys())
    user = supabase.table(TABLE_USER).select(columns).eq("id", user_uuid).execute()
    if not user.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    supabase.table(TABLE_PREDICTION).delete().eq("user_id", user_uuid).execute()
    return {"detail": "User prediction history deleted successfully"}