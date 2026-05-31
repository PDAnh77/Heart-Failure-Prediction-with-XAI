import re
import uuid
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy import delete, insert, select
from models.user_model import User
from models.prediction_model import Prediction
from models.batch_prediction_model import BatchPrediction
from schemas.patient_schema import PatientPredict
from sqlalchemy.orm import Session
from core.model_loader import get_pipeline
from services.xai_service import generate_patient_xai_images, generate_batch_xai_images
from services.dataset_service import load_dataset
from core.supabase_client import supabase

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
COLUMN_ALIASES = {
    "age": ["age", "patient age", "years", "age_years"],
    "sex": ["sex", "gender", "patient sex", "biological sex"],
    "chest_pain_type": ["chest pain type", "cp_type", "cp", "cp type", "chestpain"],
    "resting_bp": ["restingbp", "resting bp", "resting blood pressure", "rest bp"],
    "cholesterol": ["chol", "cholesterol", "cholesterol level", "serum cholesterol", "cholesterol mg/dl"],
    "fasting_bs": ["fasting blood sugar", "fasting_bs", "fasting bs", "fbs", "fasting glucose"],
    "resting_ecg": ["resting_ecg", "resting ecg", "rest ecg", "electrocardiogram", "resting electrocardiogram"],
    "max_hr": ["maxhr", "max hr", "max heart rate", "maximum heart rate", "peak hr"],
    "exercise_angina": ["exercise angina", "exercise induced angina", "exang"],
    "oldpeak": ["oldpeak", "old peak", "st depression", "st_depression"],
    "st_slope": ["st_slope", "st slope", "stslope", "st segment slope"],
}
REQUIRED_COLUMNS = list(RENAME_MAP.keys())
DATASET_BUCKET = "heart-failure-datasets"


def normalize_col(col: str) -> str:
    # Lowercase + remove space + underscore
    return re.sub(r"[\s_]+", "", col.strip().lower())


def build_column_mapping(df_columns: List[str]) -> Dict[str, str]:
    """
    Map actual dataframe columns -> REQUIRED_COLUMNS
    using normalize + alias
    """
    normalized_df_cols = {normalize_col(col): col for col in df_columns}

    mapping = {}
    missing_columns = []

    for required_col in REQUIRED_COLUMNS:
        norm_required = normalize_col(required_col)

        # 1. Exact match
        if norm_required in normalized_df_cols:
            mapping[normalized_df_cols[norm_required]] = required_col
            continue

        # 2. Alias match
        aliases = COLUMN_ALIASES.get(required_col, [])
        found = False

        for alias in aliases:
            norm_alias = normalize_col(alias)
            if norm_alias in normalized_df_cols:
                mapping[normalized_df_cols[norm_alias]] = required_col
                found = True
                break

        if not found:
            missing_columns.append(required_col)

    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Uploaded file is missing required columns", "missing_columns": missing_columns},
        )

    return mapping


def preprocess(df_input, pipeline):
    label_encoders = pipeline["label_encoders"]
    scalers = pipeline["scalers"]

    # Encode categorical columns
    for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
        df_input[col] = label_encoders[col].transform(df_input[col])

    # Scale numeric columns
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

    total = len(predictions)

    normal_count = int(np.sum(predictions == 0))
    disease_count = int(np.sum(predictions == 1))

    normal_ratio = round(normal_count / total, 4)
    disease_ratio = round(disease_count / total, 4)

    results = []
    for i, pred in enumerate(predictions):
        confidence = float(np.max(probs_matrix[i]))

        results.append({"patient_index": i, "prediction": int(pred), "probability": round(confidence, 4)})

    batch_plots = generate_batch_xai_images(
        model=model, background_data=background_data, processed_batch_df=x_processed
    )

    return {
        "summary": {
            "total": total,
            "normal": normal_count,
            "disease": disease_count,
            "normal_ratio": normal_ratio,
            "disease_ratio": disease_ratio,
        },
        "predictions": results,
        **batch_plots,
    }


def predict_dataframe(db: Session, dataset_id: str, user_id: str, target_column: str):
    try:
        df = load_dataset(dataset_id, user_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unable to load dataset: {str(e)}")

    if target_column and target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' not found. Available columns: {list(df.columns)}",
        )

    # Keep a copy of the original dataframe to preserve user's extra columns
    original_df = df.copy()

    # Build mapping; this automatically throws an HTTPException if required columns are missing
    column_mapping = build_column_mapping(df.columns.tolist())

    # Rename columns to standard names
    df_model = df.rename(columns=column_mapping)

    # Filter only required columns and replace NaN with None for Pydantic validation
    df_model = df_model[REQUIRED_COLUMNS].copy()
    df_model = df_model.where(pd.notnull(df_model), None)

    records = df_model.to_dict(orient="records")

    valid_patients: List[PatientPredict] = []
    validation_errors = []

    # Validate each row using Pydantic
    for index, row_dict in enumerate(records):
        try:
            patient = PatientPredict(**row_dict)
            valid_patients.append(patient)

        except ValidationError as e:
            for err in e.errors():
                field_name = err.get("loc")[0]
                error_msg = err.get("msg")
                # Actual data row = index + 2 (header skip)
                validation_errors.append(f"Row {index + 2} - Column '{field_name}': {error_msg}")

    if validation_errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": "Invalid data in the dataset. Please check again.", "errors": validation_errors},
        )

    # Run batch prediction
    batch_result = predict_batch(valid_patients)

    # Extract prediction results and probabilities
    predictions_list = [item["prediction"] for item in batch_result["predictions"]]
    probabilities_list = [item["probability"] for item in batch_result["predictions"]]

    # Append to the original dataframe
    result_col_name = target_column if target_column else "prediction_result"
    prob_col_name = f"{result_col_name}_prediction_probability" if target_column else "prediction_probability"

    original_df[result_col_name] = predictions_list
    original_df[prob_col_name] = probabilities_list

    pred_dataset_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
    path = f"batch-prediction/{user_id}/{pred_dataset_id}.csv"

    file_bytes = original_df.to_csv(index=False).encode("utf-8")

    # Upload the file to Supabase storage
    try:
        supabase.storage.from_(DATASET_BUCKET).upload(
            path=path, file=file_bytes, file_options={"content-type": "text/csv"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction completed, but failed to save result file: {str(e)}",
        )

    batch_xai_data = {k: v for k, v in batch_result.items() if k not in ["summary", "predictions"]}

    new_batch_pred = db.execute(
        insert(BatchPrediction)
        .values(
            user_id=check_uuid(user_id),
            source_dataset_id=dataset_id,
            result_file_id=pred_dataset_id,
            summary=batch_result["summary"],
            batch_xai=batch_xai_data,
        )
        .returning(BatchPrediction.id, BatchPrediction.created_at)
    )

    new_row = new_batch_pred.one()
    db.commit()

    # Remove 'predictions' array from the final response
    batch_result.pop("predictions", None)

    batch_result["file_id"] = pred_dataset_id
    batch_result["batch_prediction_id"] = new_row.id
    batch_result["created_at"] = new_row.created_at
    return batch_result


def check_uuid(id: str):
    try:
        validated_uuid = str(uuid.UUID(id))
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


def get_batch_predictions_by_user(db: Session, user_id: str, limit: int, offset: int):
    user_uuid = check_uuid(user_id)
    result = (
        db.execute(
            select(BatchPrediction)
            .where(BatchPrediction.user_id == user_uuid)
            .order_by(BatchPrediction.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        .scalars()
        .all()
    )
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch predictions not found")
    return result


def get_batch_prediction_by_id(db: Session, batch_id: str, user: dict):
    batch_uuid = check_uuid(batch_id)

    if user["role"] == "admin":
        result = db.execute(select(BatchPrediction).where(BatchPrediction.id == batch_uuid)).scalar_one_or_none()
    else:
        result = db.execute(
            select(BatchPrediction)
            .where(BatchPrediction.id == batch_uuid)
            .where(BatchPrediction.user_id == check_uuid(user["user_id"]))
        ).scalar_one_or_none()

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch prediction not found")
    return result


def generate_single_xai(patient_raw: Dict[str, Any]):
    pipeline = get_pipeline()
    model = pipeline["model"]
    features = pipeline["features"]
    background_data = pipeline["shap_background"]
    lime_data = pipeline["lime_training_data"]

    # Chuyển raw data thành DataFrame để tận dụng hàm build_column_mapping
    raw_df = pd.DataFrame([patient_raw])

    # Map tên cột (hàm này có sẵn trong file của bạn, dùng biến COLUMN_ALIASES)
    try:
        column_mapping = build_column_mapping(raw_df.columns.tolist())
    except HTTPException as e:
        raise e

    mapped_df = raw_df.rename(columns=column_mapping)

    # Lọc lấy đúng các cột cần thiết (REQUIRED_COLUMNS)
    mapped_df = mapped_df[REQUIRED_COLUMNS].copy()
    mapped_df = mapped_df.where(pd.notnull(mapped_df), None)

    # Chuyển lại thành dict để đưa cho Pydantic
    mapped_record = mapped_df.to_dict(orient="records")[0]

    # Validate bằng PatientPredict
    try:
        mapped_record["save_prediction"] = False
        patient_valid = PatientPredict(**mapped_record)
    except ValidationError as e:
        error_details = [f"{err.get('loc')[0]}: {err.get('msg')}" for err in e.errors()]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": "Invalid patient data format", "errors": error_details},
        )

    # Bỏ các trường không nằm trong mô hình dự đoán
    clean_dict = patient_valid.model_dump()
    clean_dict.pop("save_prediction", None)

    clean_df = pd.DataFrame([clean_dict])
    x_processed = prepare_dataframe(clean_df, pipeline)

    plots = generate_patient_xai_images(
        model=model,
        background_data=background_data,
        lime_train_data=lime_data,
        features_list=features,
        processed_df=x_processed,
        raw_row=clean_df[features],
    )
    return plots


def delete_prediction_by_id(db: Session, prediction_id: str, user: dict):
    prediction_uuid = check_uuid(prediction_id)

    stmt = delete(Prediction).where(Prediction.id == prediction_uuid)

    if user["role"] != "admin":
        stmt = stmt.where(Prediction.user_id == user["user_id"])

    result = db.execute(stmt)
    db.commit()

    if result.rowcount <= 0:
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
