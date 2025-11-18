import joblib
import pandas as pd
from db.database import supabase

# Load model khi service khởi động
BUCKET_NAME = "heart-prediction-models"
MODEL_FILENAME = "model_lr.pkl"
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILENAME}"

def load_pipeline():
    print(f"Connecting Storage to download {MODEL_FILENAME}...")
    try:
        # Download file from bucket (returns bytes)
        data = supabase.storage.from_(BUCKET_NAME).download(MODEL_FILENAME)
        
        # Write bytes to a physical file at /tmp
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(data)
            
        print(f"Download complete.")
        loaded_pipeline = joblib.load(LOCAL_MODEL_PATH)
        
        return loaded_pipeline
    except Exception as e:
        print(f"Unable to download model from Supabase.")
        print(f"Error details: {str(e)}")
        raise e

pipeline = load_pipeline()
print("Model loaded successfully.")

# # Load model khi service khởi động (local)
# import os
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(BASE_DIR, "../models/model_lr.pkl")
# pipeline = joblib.load(model_path)

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

def preprocess(data):
    if isinstance(data, list):
        # 1. Nếu là list (batch) -> tạo DataFrame N hàng
        df = pd.DataFrame(data)
    else:
        # 2. Nếu là dict (single) -> tạo DataFrame 1 hàng
        df = pd.DataFrame([data])

    df.rename(columns=RENAME_MAP, inplace=True)

    label_encoders = pipeline['label_encoders']
    scalers = pipeline['scalers']

    # Encode
    for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        df[col] = label_encoders[col].transform(df[col])

    # Scale
    df['Oldpeak'] = scalers['MinMax_Oldpeak'].transform(df[['Oldpeak']])
    df[['Age','RestingBP','Cholesterol','MaxHR']] = scalers['Standard_Numeric'].transform(df[['Age','RestingBP','Cholesterol','MaxHR']])

    return df

def predict_result(patient_data):
    model = pipeline['model']
    features = pipeline['features']
    background_data = pipeline['background_data']

    df = preprocess(patient_data)
    x = df[features]
    prediction = model.predict(x.values)
    if isinstance(patient_data, list):
        return prediction.tolist()
    else:
        return int(prediction[0])
