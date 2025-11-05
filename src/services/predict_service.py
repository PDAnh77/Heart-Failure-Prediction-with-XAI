import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Load model khi service khởi động
model, features, target = joblib.load("models/model_lr.pkl")

# Chuẩn bị các encoder / scaler
le = LabelEncoder()
mms = MinMaxScaler()
ss = StandardScaler()

def preprocess(data_dict):
    df = pd.DataFrame([data_dict])

    rename_map = {
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
    df.rename(columns=rename_map, inplace=True)

    # Encode
    for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        df[col] = le.fit_transform(df[col])

    # Scale
    df['Oldpeak'] = mms.fit_transform(df[['Oldpeak']])
    df['Age'] = ss.fit_transform(df[['Age']])
    df['RestingBP'] = ss.fit_transform(df[['RestingBP']])
    df['Cholesterol'] = ss.fit_transform(df[['Cholesterol']])
    df['MaxHR'] = ss.fit_transform(df[['MaxHR']])

    return df

def predict_result(patient_data):
    df = preprocess(patient_data)
    x = df[features]
    prediction = model.predict(x.values)
    return int(prediction[0])
