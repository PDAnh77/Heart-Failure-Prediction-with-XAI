from fastapi import HTTPException
from db.database import supabase
from schemas.prediction_schema import PredictionBase, PredictionGet

TABLE_NAME_PREDICTION = "prediction_histories"

def get_user_predictions_service(user_id: str):
    columns = ','.join(PredictionBase.model_fields.keys())
    result = supabase.table(TABLE_NAME_PREDICTION).select(columns).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")
    return result.data

def get_info_prediction_service(prediction_id: str):
    columns = ','.join(PredictionGet.model_fields.keys())
    result = supabase.table(TABLE_NAME_PREDICTION).select(columns).eq("id", prediction_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result.data

def delete_prediction_service(prediction_id: str):
    result = supabase.table(TABLE_NAME_PREDICTION).delete().eq("id", prediction_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"detail": "Delete prediction history successfully"}

def delete_user_predictions_service(user_id: str):
    result = supabase.table(TABLE_NAME_PREDICTION).delete().eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="User not found")
    return result.data