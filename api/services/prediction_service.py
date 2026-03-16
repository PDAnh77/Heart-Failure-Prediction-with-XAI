from typing import List
from uuid import UUID
import pandas as pd
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy import delete, insert, select
from models.user_model import User
from models.prediction_model import Prediction
from schemas.patient_schema import PatientPredict
from sqlalchemy.orm import Session
from core.model_loader import get_pipeline
from services.xai_service import generate_patient_xai_images, generate_batch_xai_images

RENAME_MAP = {
    "age": "Age",
    "sex": "Sex",
    "chest_pain_type": "ChestPainType",
    "resting_bp": "RestingBP",
    "cholesterol": "Cholesterol",
    "fasting_bs": "FastingBS",
    "resting_ecg": "RestingECG",
    "max_hr": "MaxHR",
    "exercise_angina": "ExerciseAngina",
    "oldpeak": "Oldpeak",
    "st_slope": "ST_Slope",
}


def preprocess(df_input, pipeline):
    label_encoders = pipeline["label_encoders"]
    scalers = pipeline["scalers"]

    # Encode các cột categorical
    for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
        df_input[col] = label_encoders[col].transform(df_input[col])

    # Scale các cột numeric
    df_input["Oldpeak"] = scalers["MinMax_Oldpeak"].transform(df_input[["Oldpeak"]])
    df_input[["Age", "RestingBP", "Cholesterol", "MaxHR"]] = scalers["Standard_Numeric"].transform(
        df_input[["Age", "RestingBP", "Cholesterol", "MaxHR"]]
    )

    return df_input


def prepare_dataframe(df: pd.DataFrame, pipeline):
    df.rename(columns=RENAME_MAP, inplace=True)
    df_processed = preprocess(df.copy(), pipeline)

    features = pipeline["features"]
    return df_processed[features]


def predict_single(db: Session, patient: PatientPredict, user_id: str):
    pipeline = get_pipeline()
    model = pipeline["model"]
    features = pipeline["features"]
    background_data = pipeline["shap_background"]
    lime_data = pipeline["lime_training_data"]

    patient_data = patient.model_dump()
    save_prediction = patient_data.pop("save_prediction")

    raw_df = pd.DataFrame([patient_data])

    x_processed = prepare_dataframe(raw_df, pipeline)

    predictions = model.predict(x_processed.values)
    probs_matrix = model.predict_proba(x_processed.values)

    pred = predictions[0]
    confidence = float(np.max(probs_matrix[0]))

    plots = generate_patient_xai_images(
        model=model,
        background_data=background_data,
        lime_train_data=lime_data,
        features_list=features,
        processed_df=x_processed,
        raw_row=raw_df[features],
    )

    results = {"prediction": int(pred), "probability": round(confidence, 4), **plots}

    if save_prediction:
        new_prediction = db.execute(
            insert(Prediction)
            .values(
                user_id=user_id,
                input_data=patient_data,
                prediction_xai=plots,
                predicted_label=int(pred),
                predicted_probability=round(confidence, 4),
            )
            .returning(Prediction.id, Prediction.created_at)
        )

        new_row = new_prediction.one()
        db.commit()

        results["prediction_history"] = {
            "id": new_row.id,
            "created_at": new_row.created_at,
        }
    return results


def predict_batch(patients: List[PatientPredict]):
    pipeline = get_pipeline()
    model = pipeline["model"]
    background_data = pipeline["shap_background"]

    patient_data_list = [patient.model_dump() for patient in patients]

    raw_df = pd.DataFrame(patient_data_list)

    x_processed = prepare_dataframe(raw_df, pipeline)

    predictions = model.predict(x_processed.values)
    probs_matrix = model.predict_proba(x_processed.values)

    results = []
    for i, pred in enumerate(predictions):
        confidence = float(np.max(probs_matrix[i]))

        results.append({"patient_index": i, "prediction": int(pred), "probability": round(confidence, 4)})

    batch_plots = {}
    batch_plots = generate_batch_xai_images(
        model=model, background_data=background_data, processed_batch_df=x_processed
    )

    return {"predictions": results, **batch_plots}


def check_uuid(id: str):
    try:
        validated_uuid = str(UUID(id))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format")
    return validated_uuid


def get_predictions_by_user(db: Session, user_id: str, limit: int, offset: int):
    user_uuid = check_uuid(user_id)
    result = (
        db.execute(
            select(Prediction)
            .where(Prediction.user_id == user_uuid)
            .order_by(Prediction.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        .scalars()
        .all()
    )
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return result


def get_prediction_by_id(db: Session, prediction_id: str, user: dict):
    prediction_uuid = check_uuid(prediction_id)

    if user["role"] == "admin":
        result = db.execute(select(Prediction).where(Prediction.id == prediction_uuid)).scalar_one_or_none()
    else:
        result = db.execute(
            select(Prediction).where(Prediction.id == prediction_uuid).where(Prediction.user_id == user["user_id"])
        ).scalar_one_or_none()

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return result


def delete_prediction_by_id(db: Session, prediction_id: str, user: dict):
    prediction_uuid = check_uuid(prediction_id)

    stmt = delete(Prediction).where(Prediction.id == prediction_uuid)

    if user["role"] != "admin":
        stmt = stmt.where(Prediction.user_id == user["user_id"])

    result = db.execute(stmt)
    db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")

    return {"detail": "Delete user prediction successfully"}


def delete_predictions_by_user(db: Session, user_id: str):
    user_uuid = check_uuid(user_id)

    user = db.execute(select(User.id).where(User.id == user_uuid)).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    db.execute(delete(Prediction).where(Prediction.user_id == user_uuid))
    db.commit()
    return {"detail": "User predictions deleted successfully"}
