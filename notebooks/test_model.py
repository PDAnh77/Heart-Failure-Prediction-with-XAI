import joblib
import shap
import pandas as pd
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("../data/heart.csv")
new_sample = data.sample(50)
# new_sample = pd.DataFrame([
#     [47, 'F', 'NAP', 132, 245, 0, 'Normal', 158, 'N', 0.6, 'Up', 0],
#     [61, 'M', 'ASY', 150, 198, 1, 'LVH', 129, 'Y', 2.3, 'Flat', 1],
#     [52, 'F', 'ATA', 118, 220, 0, 'Normal', 166, 'N', 0.0, 'Up', 0],
#     [67, 'M', 'TA', 170, 270, 1, 'ST', 111, 'Y', 1.5, 'Down', 1],
#     [43, 'M', 'NAP', 126, 305, 0, 'LVH', 172, 'N', 0.4, 'Up', 0],
#     [55, 'F', 'ASY', 140, 233, 1, 'ST', 137, 'Y', 3.2, 'Flat', 1],
#     [49, 'M', 'ATA', 120, 189, 0, 'Normal', 172, 'N', 0.2, 'Up', 0],
#     [62, 'F', 'NAP', 134, 268, 0, 'LVH', 160, 'N', 0.5, 'Flat', 1],
#     [58, 'M', 'TA', 146, 251, 1, 'ST', 112, 'Y', 2.6, 'Down', 1],
#     [45, 'F', 'ATA', 128, 210, 0, 'Normal', 175, 'N', 0.1, 'Up', 0]
# ], columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
#             'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease'])

pipeline = joblib.load("../src/models/model_lr.pkl")

model = pipeline['model']                       # classifier LogisticRegression
label_encoders = pipeline['label_encoders']    # dict các LabelEncoder
scalers = pipeline['scalers']                  # dict các scaler
features = pipeline['features']                # Index object hoặc list cột
target = pipeline['target']                    # tên cột target
background_data = pipeline['background_data']

df = new_sample.copy(deep = True)

# Label encoding - Biến dữ liệu dạng chữ (categorical) → dạng số (integer).
df['Sex'] = label_encoders['Sex'].transform(df['Sex'])
df['ChestPainType'] = label_encoders['ChestPainType'].transform(df['ChestPainType'])
df['RestingECG'] = label_encoders['RestingECG'].transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoders['ExerciseAngina'].transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoders['ST_Slope'].transform(df['ST_Slope'])

# Feature Scaling (Normalization / Standardization) - Chuẩn hóa dữ liệu số (numeric) về cùng thang đo
df['Oldpeak'] = scalers['MinMax_Oldpeak'].transform(df[['Oldpeak']])
df[['Age','RestingBP','Cholesterol','MaxHR']] = scalers['Standard_Numeric'].transform(df[['Age','RestingBP','Cholesterol','MaxHR']])

x_new_processed = df[features]
predictions = model.predict(x_new_processed.values)

# y_new = df[target]
# print("Accuracy : ",'{0:.2%}'.format(accuracy_score(y_new,predictions)))

explainer = shap.LinearExplainer(model, background_data.data)
shap_values = explainer(x_new_processed)
shap_values.display_data = new_sample[features].values

i = 0 # sample
print(f"Giải thích cho bệnh nhân thứ {i}: {new_sample.iloc[i]['Age']} tuổi, Giới tính {new_sample.iloc[0]['Sex']}")
shap.plots.waterfall(shap_values[i])
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)