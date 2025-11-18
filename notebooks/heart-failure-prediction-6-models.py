import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from lime.lime_tabular import LimeTabularExplainer
pd.options.display.float_format = '{:.2f}'.format
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# ----------------------------------------------------------
# Load and preprocess
# ----------------------------------------------------------
data = pd.read_csv('../data/heart.csv')

df1 = data.copy(deep=True)

# Label Encoding
le_sex = LabelEncoder()
le_chest = LabelEncoder()
le_ecg = LabelEncoder()
le_angina = LabelEncoder()
le_slope = LabelEncoder()

# Fit và transform từng cột
le_sex.fit(df1['Sex'])
df1['Sex'] = le_sex.transform(df1['Sex'])

le_chest.fit(df1['ChestPainType'])
df1['ChestPainType'] = le_chest.transform(df1['ChestPainType'])

le_ecg.fit(df1['RestingECG'])
df1['RestingECG'] = le_ecg.transform(df1['RestingECG'])

le_angina.fit(df1['ExerciseAngina'])
df1['ExerciseAngina'] = le_angina.transform(df1['ExerciseAngina'])

le_slope.fit(df1['ST_Slope'])
df1['ST_Slope'] = le_slope.transform(df1['ST_Slope'])

# MinMaxScaler cho Oldpeak
mms = MinMaxScaler()
df1['Oldpeak'] = mms.fit_transform(df1[['Oldpeak']])

# StandardScaler cho numerical
std_cols = ['Age','RestingBP','Cholesterol','MaxHR']
ss = StandardScaler()
df1[std_cols] = ss.fit_transform(df1[std_cols])

# Drop RestingBP, RestingECG
features = df1[df1.columns.drop(['HeartDisease','RestingBP','RestingECG'])]
target = df1['HeartDisease']

# Giữ nguyên DataFrame → SHAP đọc được tên cột
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=2)

colors = ['#F93822','#FDD20E']


# ----------------------------------------------------------
# Model wrapper (train bằng numpy, SHAP bằng DataFrame)
# ----------------------------------------------------------
def model(classifier):
    classifier.fit(x_train.values, y_train.values)
    prediction = classifier.predict(x_test.values)
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    print('Accuracy: ', '{0:.2%}'.format(accuracy_score(y_test, prediction)))
    print('Cross Validation Score: ', '{0:.2%}'.format(
        cross_val_score(classifier, x_train.values, y_train.values, cv=cv, scoring='roc_auc').mean()
    ))
    print('ROC_AUC Score: ', '{0:.2%}'.format(roc_auc_score(y_test, prediction)))
    
    RocCurveDisplay.from_estimator(classifier, x_test.values, y_test)
    plt.title('ROC_AUC_Plot')
    plt.show()


def model_evaluation(classifier):
    cm = confusion_matrix(y_test, classifier.predict(x_test.values))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, cmap=colors, fmt='')
    plt.show()
    
    print(classification_report(y_test, classifier.predict(x_test.values)))

# ----------------------------------------------------------
# Train models
# ----------------------------------------------------------
classifier_lr = LogisticRegression(random_state=0, C=10, penalty='l2')
model(classifier_lr)
model_evaluation(classifier_lr)

classifier_svc = SVC(kernel = 'linear',C = 0.1)
model(classifier_svc)
model_evaluation(classifier_svc)

classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)
model(classifier_dt) 
model_evaluation(classifier_dt)

classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)
model(classifier_rf)
model_evaluation(classifier_rf)

classifier_knn = KNeighborsClassifier(leaf_size = 1, n_neighbors = 3,p = 1)
model(classifier_knn)
model_evaluation(classifier_knn)

classifier_xgb = XGBClassifier(
    random_state=0,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss'
)
model(classifier_xgb)
model_evaluation(classifier_xgb)

# ----------------------------------------------------------
# SHAP (hoạt động trên DataFrame, nên có tên cột)
# ----------------------------------------------------------
i = 0 # sample
shap_explainer = shap.Explainer(classifier_lr, x_train)    # x_train là DataFrame
shap_values = shap_explainer(x_test)                       # x_test là DataFrame

# Plot SHAP
shap.plots.waterfall(shap_values[i])
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)

background_summary = shap.kmeans(x_train, 50)

# joblib.dump(
#     {
#         "model": classifier_lr,
#         "label_encoders": {
#             "Sex": le_sex,         # LabelEncoder cho từng cột phân loại
#             "ChestPainType": le_chest,
#             "RestingECG": le_ecg,
#             "ExerciseAngina": le_angina,
#             "ST_Slope": le_slope
#         },
#         "scalers": {
#             "MinMax_Oldpeak": mms,
#             "Standard_Numeric": ss
#         },
#         "features": features.columns,
#         "target": "HeartDisease",
#         "background_data": background_summary  # Lưu bản tóm tắt
#     },
#     "../src/models/model_lr.pkl"
# )