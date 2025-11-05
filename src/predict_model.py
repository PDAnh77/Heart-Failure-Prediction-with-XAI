import joblib
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# Load data
data = pd.read_csv("../data/heart.csv")
# new_sample = data.sample(50)
new_sample = pd.DataFrame([
    [47, 'F', 'NAP', 132, 245, 0, 'Normal', 158, 'N', 0.6, 'Up', 0],
    [61, 'M', 'ASY', 150, 198, 1, 'LVH', 129, 'Y', 2.3, 'Flat', 1],
    [52, 'F', 'ATA', 118, 220, 0, 'Normal', 166, 'N', 0.0, 'Up', 0],
    [67, 'M', 'TA', 170, 270, 1, 'ST', 111, 'Y', 1.5, 'Down', 1],
    [43, 'M', 'NAP', 126, 305, 0, 'LVH', 172, 'N', 0.4, 'Up', 0],
    [55, 'F', 'ASY', 140, 233, 1, 'ST', 137, 'Y', 3.2, 'Flat', 1],
    [49, 'M', 'ATA', 120, 189, 0, 'Normal', 172, 'N', 0.2, 'Up', 0],
    [62, 'F', 'NAP', 134, 268, 0, 'LVH', 160, 'N', 0.5, 'Flat', 1],
    [58, 'M', 'TA', 146, 251, 1, 'ST', 112, 'Y', 2.6, 'Down', 1],
    [45, 'F', 'ATA', 128, 210, 0, 'Normal', 175, 'N', 0.1, 'Up', 0]
], columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
            'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease'])

model, features, target = joblib.load("./models/model_lr.pkl")

# Label encoding - Biến dữ liệu dạng chữ (categorical) → dạng số (integer).
le = LabelEncoder()
df1 = new_sample.copy(deep = True)

df1['Sex'] = le.fit_transform(df1['Sex'])
df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])

# Feature Scaling (Normalization / Standardization) - Chuẩn hóa dữ liệu số (numeric) về cùng thang đo
mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

df1['Oldpeak'] = mms.fit_transform(df1[['Oldpeak']])
df1['Age'] = ss.fit_transform(df1[['Age']])
df1['RestingBP'] = ss.fit_transform(df1[['RestingBP']])
df1['Cholesterol'] = ss.fit_transform(df1[['Cholesterol']])
df1['MaxHR'] = ss.fit_transform(df1[['MaxHR']])

# Load model

# print(model.coef_)       # nếu ra mảng số → đã train
# print(model.intercept_)

x_new = df1[features]
y_new = df1[target]
predictions = model.predict(x_new.values)

print("Accuracy : ",'{0:.2%}'.format(accuracy_score(y_new,predictions)))