import joblib
from matplotlib import pyplot as plt
import shap
import pandas as pd
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer

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

pipeline = joblib.load("../src/models/model_predict.pkl")

model = pipeline['model']                      # classifier LogisticRegression
label_encoders = pipeline['label_encoders']    # dict các LabelEncoder
scalers = pipeline['scalers']                  # dict các scaler
features = pipeline['features']                # Index object hoặc list cột
target = pipeline['target']                    # tên cột target
background_data = pipeline['shap_background']
lime_data = pipeline['lime_training_data']

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

explainer = shap.Explainer(model, background_data.data)
shap_values = explainer(x_new_processed)
shap_values.display_data = new_sample[features].values

i = 0 # sample
print(f"Giải thích cho bệnh nhân thứ {i}:")
shap.plots.waterfall(shap_values[i], show=False)
plt.title(f"Individual Prediction Explanation", fontsize=16)  
plt.show()

shap.plots.bar(shap_values, show=False)
plt.title("Local Feature Importance Ranking", fontsize=16)
plt.show()

shap.plots.beeswarm(shap_values, show=False)
plt.title("Global Feature Impact Distribution", fontsize=16)
plt.show()

explainer = LimeTabularExplainer(
    training_data=lime_data,
    feature_names=list(features),
    class_names=['Normal', 'Heart Disease'],
    mode='classification'
)

exp = explainer.explain_instance(
    data_row=x_new_processed.iloc[i].values,
    predict_fn=model.predict_proba
)

fig = exp.as_pyplot_figure()
plt.show()