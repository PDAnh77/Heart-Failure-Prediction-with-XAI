import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import lightgbm as lgb
from lime.lime_tabular import LimeTabularExplainer

pd.options.display.float_format = "{:.2f}".format
import warnings

warnings.filterwarnings("ignore")

# --- Thư viện mới cho Missing Data Imputation ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from imblearn.over_sampling import ADASYN
from collections import Counter

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# Copy từ notebooks/heart-failure-prediction-6-models.py

# ----------------------------------------------------------
# Load and preprocess
# ----------------------------------------------------------
data = pd.read_csv("../input/heart-failure-prediction/heart.csv")
df1 = data.copy(deep=True)

# ==========================================================
# 0. XỬ LÝ NGOẠI LAI (OUTLIER) BẰNG PHƯƠNG PHÁP IQR
# ==========================================================
print("\n--- BẮT ĐẦU PHÁT HIỆN VÀ XỬ LÝ NGOẠI LAI (IQR) ---")

# Chỉ áp dụng IQR cho các cột số liên tục có ý nghĩa lâm sàng
# Loại trừ: FastingBS (binary 0/1), HeartDisease (target)
# Lưu ý đặc biệt:
#   - Cholesterol=0: là missing data ẩn (mã hóa bằng 0), KHÔNG xử lý tại đây
#   - RestingBP=0: bất khả thi về sinh học → xử lý riêng dướbên i
#   - Oldpeak âm: bất hợp lý lâm sàng → clip về 0
numeric_outlier_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

print("\n[IQR] Thống kê TRƯỚC khi xử lý:")
print(df1[numeric_outlier_cols].describe().T[["min", "25%", "50%", "75%", "max"]])

# --- BƯỚC 0.1: Xử lý đặc biệt trước IQR ---

# RestingBP = 0 là bất khả thi sinh học → thay bằng NaN để impute sau
bp_zero_mask = df1["RestingBP"] == 0
if bp_zero_mask.sum() > 0:
    print(f"\n[RestingBP] Phát hiện {bp_zero_mask.sum()} giá trị = 0 (bất khả thi) → thay bằng NaN")
    df1.loc[bp_zero_mask, "RestingBP"] = np.nan

# Cholesterol = 0 là missing data ẩn → chuyển sang NaN để MICE xử lý sau
chol_zero_mask = df1["Cholesterol"] == 0
if chol_zero_mask.sum() > 0:
    print(f"[Cholesterol] Phát hiện {chol_zero_mask.sum()} giá trị = 0 (missing ẩn) → thay bằng NaN")
    df1.loc[chol_zero_mask, "Cholesterol"] = np.nan


# --- BƯỚC 0.2: Phát hiện outlier bằng IQR và Winsorization ---
# Chiến lược: KHÔNG xóa hàng (dataset nhỏ, 918 mẫu, dữ liệu y tế quý)
# Thay vào đó: clip về fence [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (Winsorization)

outlier_summary = {}
for col in numeric_outlier_cols:
    col_data = df1[col].dropna()
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    n_lower = (df1[col] < lower_fence).sum()
    n_upper = (df1[col] > upper_fence).sum()
    n_total = n_lower + n_upper

    outlier_summary[col] = {
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "Lower Fence": lower_fence, "Upper Fence": upper_fence,
        "Outliers (dưới)": n_lower, "Outliers (trên)": n_upper, "Tổng outlier": n_total
    }

    if n_total > 0:
        # Winsorization: clip về fence thay vì xóa
        df1[col] = df1[col].clip(lower=lower_fence, upper=upper_fence)

# In bảng tổng hợp kết quả IQR
print("\n[IQR] Kết quả phát hiện outlier:")
summary_df = pd.DataFrame(outlier_summary).T
print(summary_df[["Q1", "Q3", "IQR", "Lower Fence", "Upper Fence",
                   "Outliers (dưới)", "Outliers (trên)", "Tổng outlier"]].to_string())

print("\n[IQR] Thống kê SAU khi xử lý (Winsorization):")
print(df1[numeric_outlier_cols].describe().T[["min", "25%", "50%", "75%", "max"]])
print("--- HOÀN THÀNH XỬ LÝ NGOẠI LAI ---\n")

# ==========================================================
# GIẢ LẬP DỮ LIỆU KHUYẾT CHO 2 CỘT: CHOLESTEROL (4% missing) & OLDPEAK (10% missing)
# ==========================================================
print("\n--- BẮT ĐẦU GIẢ LẬP DỮ LIỆU KHUYẾT (CHOLESTEROL & OLDPEAK) ---")
np.random.seed(42) # Cố định seed để dễ debug

# 1. Nhóm thiếu <= 5% (Test Mean Imputation)
# 'Cholesterol' cho trống 4%
missing_indices_chol = df1.sample(frac=0.04, random_state=42).index
df1.loc[missing_indices_chol, 'Cholesterol'] = np.nan

# 2. Nhóm thiếu > 5% (Test MICE Imputation)
# 'Oldpeak' cho trống 10%
# Dùng random_state khác để các dòng bị thiếu của Oldpeak không trùng hoàn toàn với Cholesterol
missing_indices_oldpeak = df1.sample(frac=0.10, random_state=3).index 
df1.loc[missing_indices_oldpeak, 'Oldpeak'] = np.nan

print("Tổng hợp số lượng ô bị khuyết vừa tạo ra:")
print(df1.isnull().sum()[df1.isnull().sum() > 0])

# ----------------------------------------------------------
# 1. TÁCH TẬP DỮ LIỆU (SPLITTING) TRƯỚC TIÊN — tránh Data Leakage
# ----------------------------------------------------------
print("--- Chia tập Train/Test trước khi tiền xử lý ---")

features = df1.drop(columns=["HeartDisease", "RestingBP", "RestingECG"])
target = df1["HeartDisease"]

# stratify=target giữ nguyên tỷ lệ nhãn giữa hai tập
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.20, random_state=2, stratify=target
)

# ----------------------------------------------------------
# 2. MISSING DATA IMPUTATION (fit trên Train, transform trên Test)
# ----------------------------------------------------------
print("--- Đang xử lý Missing Data ---")

# Tính tỷ lệ missing chỉ trên tập Train
missing_percentages = x_train.isnull().mean() * 100
cols_missing_under_5 = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 5)].index.tolist()
cols_missing_over_5 = missing_percentages[missing_percentages > 5].index.tolist()

# BƯỚC 2.1: Xử lý các cột thiếu <= 5% (Mean cho Numeric, Mode cho Categorical)
for col in cols_missing_under_5:
    if x_train[col].dtype in ['int64', 'float64']:
        imputer = SimpleImputer(strategy='mean')
    else:
        imputer = SimpleImputer(strategy='most_frequent')
    x_train[col] = imputer.fit_transform(x_train[[col]]).ravel()
    x_test[col]  = imputer.transform(x_test[[col]]).ravel()

# BƯỚC 2.2: Label Encoding — học từ điển nhãn chỉ trên Train
# Dùng OrdinalEncoder để tránh lỗi khi test set có nhãn chưa thấy trong train
categorical_cols = ["Sex", "ChestPainType", "ExerciseAngina", "ST_Slope"]
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=float)
x_train[categorical_cols] = oe.fit_transform(x_train[categorical_cols])
x_test[categorical_cols]  = oe.transform(x_test[categorical_cols])

# BƯỚC 2.3: Xử lý các cột thiếu > 5% bằng MICE (IterativeImputer)
if len(cols_missing_over_5) > 0:
    print(f"[> 5%] Áp dụng MICE Imputation cho các cột: {cols_missing_over_5}")
    mice_imputer = IterativeImputer(max_iter=10, random_state=0)
    mice_imputer.fit(x_train)
    x_train[:] = mice_imputer.transform(x_train)
    x_test[:]  = mice_imputer.transform(x_test)

# ----------------------------------------------------------
# 3. SCALING (fit trên Train, transform trên Test)
# ----------------------------------------------------------
# MinMaxScaler cho Oldpeak
mms = MinMaxScaler()
x_train["Oldpeak"] = mms.fit_transform(x_train[["Oldpeak"]])
x_test["Oldpeak"]  = mms.transform(x_test[["Oldpeak"]])

# StandardScaler cho các cột numerical
std_cols = ["Age", "Cholesterol", "MaxHR"]
ss = StandardScaler()
x_train[std_cols] = ss.fit_transform(x_train[std_cols])
x_test[std_cols]  = ss.transform(x_test[std_cols])

print("--- TIỀN XỬ LÝ HOÀN TẤT. SẴN SÀNG HUẤN LUYỆN ---")

# ----------------------------------------------------------
# 3. IMBALANCED DATA HANDLING (ADASYN)
# ----------------------------------------------------------
# print("\n--- Xử lý Imbalanced Data bằng ADASYN ---")
# print(f"Phân phối nhãn (Target) TRƯỚC khi xử lý trên tập train: {Counter(y_train)}")

# Khởi tạo thuật toán ADASYN
# adasyn = ADASYN(random_state=42)

# Fit và transform (Tạo dữ liệu tổng hợp cho nhóm thiểu số)
# LƯU Ý: Chỉ fit_resample trên tập Train để tránh Data Leakage!
# x_train_resampled, y_train_resampled = adasyn.fit_resample(x_train, y_train)

# print(f"Phân phối nhãn (Target) SAU khi xử lý trên tập train: {Counter(y_train_resampled)}")

# Chuyển đổi lại về DataFrame và Series để tương thích với các bước sau (SHAP, LIME)
# x_train = pd.DataFrame(x_train_resampled, columns=x_train.columns)
# y_train = pd.Series(y_train_resampled, name=y_train.name)

# --- SỬA LỖI NỘI SUY CỦA ADASYN & CHUẨN BỊ CHO LIGHTGBM ---
categorical_features = ['Sex', 'ChestPainType', 'ExerciseAngina', 'ST_Slope']
# Vì ADASYN nội suy dữ liệu nên các biến phân loại có thể bị chuyển thành số thập phân (VD: 0.6)
# Cần làm tròn lại về đúng số nguyên đại diện cho Category
# for col in categorical_features:
#     x_train[col] = x_train[col].round()
# Ép kiểu dữ liệu sang 'category' để kích hoạt thuật toán Categorical của LightGBM
for col in categorical_features:
    x_train[col] = x_train[col].astype('category')
    x_test[col] = x_test[col].astype('category')

# ----------------------------------------------------------

colors = ["#F93822", "#FDD20E"]
# ----------------------------------------------------------
# Model wrapper (train bằng numpy, SHAP bằng DataFrame)
# ----------------------------------------------------------
def model(classifier):
    if classifier.__class__.__name__ == 'LGBMClassifier':
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
    else:
        classifier.fit(x_train.values, y_train.values)
        prediction = classifier.predict(x_test.values)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    print("Accuracy: ", "{0:.2%}".format(accuracy_score(y_test, prediction)))
    print(
        "Cross Validation Score: ",
        "{0:.2%}".format(
            cross_val_score(
                classifier, x_train.values, y_train.values, cv=cv, scoring="roc_auc"
            ).mean()
        ),
    )
    print("ROC_AUC Score: ", "{0:.2%}".format(roc_auc_score(y_test, prediction)))

    RocCurveDisplay.from_estimator(classifier, x_test.values, y_test)
    plt.title("ROC_AUC_Plot")
    plt.show()


def model_evaluation(classifier):
    cm = confusion_matrix(y_test, classifier.predict(x_test.values))
    names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    counts = [value for value in cm.flatten()]
    percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, cmap=colors, fmt="")
    plt.show()

    print(classification_report(y_test, classifier.predict(x_test.values)))


# ----------------------------------------------------------
# Train 7 models
# ----------------------------------------------------------
classifier_lr = LogisticRegression(random_state=0, C=10, penalty="l2")
model(classifier_lr)
model_evaluation(classifier_lr)

classifier_svc = SVC(kernel="linear", C=0.1)
model(classifier_svc)
model_evaluation(classifier_svc)

classifier_dt = DecisionTreeClassifier(random_state=1000, max_depth=4, min_samples_leaf=1)
model(classifier_dt)
model_evaluation(classifier_dt)

classifier_rf = RandomForestClassifier(max_depth=4, random_state=0)
model(classifier_rf)
model_evaluation(classifier_rf)

classifier_knn = KNeighborsClassifier(leaf_size=1, n_neighbors=3, p=1)
model(classifier_knn)
model_evaluation(classifier_knn)

classifier_xgb = XGBClassifier(
    random_state=0,
    n_estimators=50,
    max_depth=3,
    learning_rate=0.105,
    subsample=0.8,
    colsample_bytree=0.9,
    eval_metric="logloss",
)
model(classifier_xgb)
model_evaluation(classifier_xgb)

lgbm_basic = lgb.LGBMClassifier(
    learning_rate=0.05,
    n_estimators=200,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    verbose=-1
)
model(lgbm_basic)
model_evaluation(lgbm_basic)

# ----------------------------------------------------------
# Local
# ----------------------------------------------------------
i = 0  # sample
# Plot SHAP
shap_explainer = shap.Explainer(classifier_xgb, x_train)
shap_values = shap_explainer(x_test)

print(f"Explanation for patient {i}:")
shap.plots.waterfall(shap_values[i], show=False)
plt.title(f"Individual Prediction Explanation", fontsize=16)
plt.show()

shap.plots.bar(shap_values[i], show=False)
plt.title("Local Feature Importance Ranking", fontsize=16)
plt.show()

# Plot LIME
explainer = LimeTabularExplainer(
    training_data=x_train.values,
    feature_names=x_train.columns,
    class_names=["Normal", "Heart Disease"],
    mode="classification",
)

exp = explainer.explain_instance(
    data_row=x_test.iloc[i].values, predict_fn=classifier_xgb.predict_proba
)

fig = exp.as_pyplot_figure()
plt.show()

# ----------------------------------------------------------
# XAI: Global Explainability
# ----------------------------------------------------------

shap.plots.bar(shap_values, show=False)
plt.title("Global Feature Importance Ranking", fontsize=16)
plt.show()

shap.plots.beeswarm(shap_values, show=False)
plt.title("Global Feature Impact Distribution", fontsize=16)
plt.show()

# Cung cấp dữ liệu tham chiếu trước khi giải thích một dự đoán cụ thể
background_summary = shap.kmeans(x_train, 50)
lime_training_data = x_train.values

# joblib.dump(
#     {
#         "model": classifier_xgb,
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
#         "shap_background": background_summary,  # Lưu bản tóm tắt
#         "lime_training_data": lime_training_data
#     },
#     "../models/model_predict.pkl"
# )
