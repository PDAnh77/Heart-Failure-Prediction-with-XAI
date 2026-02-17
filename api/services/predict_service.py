import pandas as pd
import numpy as np
from core.model_loader import get_pipeline
from services.xai_service import generate_patient_xai_images, generate_batch_xai_images
from db.database import supabase

TABLE_NAME_PREDICTION = "prediction_histories"
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
    "st_slope": "ST_Slope"
}

def preprocess(df_input, pipeline):
    label_encoders = pipeline['label_encoders']
    scalers = pipeline['scalers']

    # Encode các cột categorical
    for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        df_input[col] = label_encoders[col].transform(df_input[col])

    # Scale các cột numeric
    df_input['Oldpeak'] = scalers['MinMax_Oldpeak'].transform(df_input[['Oldpeak']])
    df_input[['Age','RestingBP','Cholesterol','MaxHR']] = scalers['Standard_Numeric'].transform(df_input[['Age','RestingBP','Cholesterol','MaxHR']])

    return df_input

def predict_result(patient_data, user_id: str):
    pipeline = get_pipeline()
    model = pipeline['model']
    features = pipeline['features']
    background_data = pipeline['shap_background']
    lime_data = pipeline['lime_training_data']

    if isinstance(patient_data, list):
        raw_df = pd.DataFrame(patient_data)
        is_batch = True
    else:
        save_prediction = patient_data.pop("save_prediction")
        raw_df = pd.DataFrame([patient_data])
        is_batch = False

    raw_df.rename(columns=RENAME_MAP, inplace=True)

    df_processed = preprocess(raw_df.copy(), pipeline)
    x_processed = df_processed[features]

    predictions = model.predict(x_processed.values)
    probs_matrix = model.predict_proba(x_processed.values)

    if is_batch:
        results = []
        for i, pred in enumerate(predictions):
            confidence = float(np.max(probs_matrix[i]))

            results.append({
                "patient_index": i,
                "prediction": int(pred),
                "probability": round(confidence, 4)
            })

        batch_plots = {}
        batch_plots = generate_batch_xai_images(
            model=model,
            background_data=background_data,
            processed_batch_df=x_processed
        )

        return {
            "predictions": results,
            **batch_plots
        }
    else:
        pred = predictions[0]
        confidence = float(np.max(probs_matrix[0]))
        
        plots = generate_patient_xai_images(
            model=model,
            background_data=background_data,
            lime_train_data=lime_data,
            features_list=features,
            processed_df=x_processed, 
            raw_row=raw_df[features]
        )

        results = {
            "prediction": int(pred),
            "probability": round(confidence, 4),
            **plots
        }

        if save_prediction:
            insert_history = supabase.table(TABLE_NAME_PREDICTION).insert({
                "user_id": user_id,
                "input_data": patient_data,
                "prediction_xai": plots,
                "predicted_label": int(pred),
                "predicted_probability": round(confidence, 4)
            }).execute()

        new_history_record = insert_history.data[0]
        results["prediction_history"] = {
            "id": new_history_record.get("id"),
            "created_at": new_history_record.get("created_at")
        }
        return results