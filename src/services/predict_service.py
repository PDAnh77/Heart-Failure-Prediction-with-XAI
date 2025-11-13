import joblib
import pandas as pd
import os

# Load model khi service khởi động
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "../models/model_lr.pkl")
pipeline = joblib.load(model_path)

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

    df = preprocess(patient_data)
    x = df[features]
    prediction = model.predict(x.values)
    if isinstance(patient_data, list):
        return prediction.tolist()
    else:
        return int(prediction[0])
