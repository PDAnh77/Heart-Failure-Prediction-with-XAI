import joblib
from matplotlib import pyplot as plt
import shap
import pandas as pd
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer

# Load data
data = pd.read_csv("../input/heart-failure-prediction/heart.csv")
new_sample = data.sample(100)
pipeline = joblib.load("../models/model_predict.pkl")

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