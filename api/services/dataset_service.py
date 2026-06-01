import io
from random import randint
import uuid
import copy
import shap
from datetime import datetime
from fastapi import HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from core.supabase_client import supabase
from core.config import settings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import IterativeImputer, KNNImputer
from imblearn.over_sampling import ADASYN, SMOTE

DATASET_BUCKET = "heart-failure-datasets"
EDA_BUCKET = "eda-artifacts"
SELECTED_MODEL = settings.FEATURE_SELECTION_MODEL
AVAILABLE_MODELS = {
    "svm": SVC(kernel="linear", C=0.1),
    "logistic_regression": LogisticRegression(random_state=0, C=10, penalty="l2"),
    "random_forest": RandomForestClassifier(max_depth=4, random_state=0),
    "decision_tree": DecisionTreeClassifier(random_state=1000, max_depth=4, min_samples_leaf=1),
    "knn": KNeighborsClassifier(leaf_size=1, n_neighbors=3, p=1),
    "xgboost": XGBClassifier(
        random_state=0,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.105,
        subsample=0.8,
        colsample_bytree=0.9,
        eval_metric="logloss",
    ),
    "lightgbm": LGBMClassifier(
        objective="binary",
        random_state=0,
        n_estimators=100,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
    ),
}


def _sanitize(obj):
    """Recursively replace nan/inf float values with None for JSON safety."""
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def load_dataset(dataset_id: str, user_id: str) -> pd.DataFrame:
    if dataset_id.startswith("pred_"):
        path = f"batch-prediction/{user_id}/{dataset_id}.csv"
    elif dataset_id.startswith("fs_"):
        path = f"feature-selection/{user_id}/{dataset_id}.csv"
    elif dataset_id.startswith("processed_"):
        path = f"processed/{user_id}/{dataset_id}.csv"
    else:
        path = f"raw/{user_id}/{dataset_id}.csv"

    try:
        # 1. Download file from storage
        response = supabase.storage.from_(DATASET_BUCKET).download(path)
    except Exception:
        # File not found / storage error
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset '{dataset_id}' not found or cannot be accessed"
        )

    try:
        # 2. Convert bytes → file-like object → DataFrame
        df = pd.read_csv(io.BytesIO(response))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to parse CSV file")

    # 3. Check empty dataset
    if df.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset is empty")
    return df


def upload_plot(plt_obj, path, bucket=EDA_BUCKET):
    img_buf = io.BytesIO()
    plt_obj.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)
    plt_obj.close()  # Giải phóng RAM

    try:
        supabase.storage.from_(bucket).upload(
            path=path, file=img_buf.getvalue(), file_options={"content-type": "image/png", "upsert": "true"}
        )
        return supabase.storage.from_(bucket).get_public_url(path)
    except Exception as e:
        print(f"Upload failed for {path}: {e}")
        return None


def upload_raw_dataset(file: UploadFile, user_id: str):
    valid_extensions = (".csv", ".xlsx", ".xls")

    filename_lower = file.filename.lower()

    if not filename_lower.endswith(valid_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV and Excel files (.xlsx, .xls) are allowed"
        )

    original_file_type = "csv"
    if filename_lower.endswith(".xlsx"):
        original_file_type = "xlsx"
    elif filename_lower.endswith(".xls"):
        original_file_type = "xls"

    # Check file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    # Read file
    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            # Đọc file Excel
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file format or corrupted file: {str(e)}"
        )

    # Dataset validation
    if df.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset is empty")

    if df.shape[0] > 100000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset too large (max 100000 rows allowed)"
        )

    if df.shape[1] > 100:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Too many columns (max 100 allowed)")

    # Extract columns
    columns = df.columns.tolist()

    # Create dataset id
    dataset_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"

    # Chuẩn hóa tất cả dữ liệu thành CSV trước khi lưu lên Supabase
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    path = f"raw/{user_id}/{dataset_id}.csv"

    # Upload
    try:
        supabase.storage.from_(DATASET_BUCKET).upload(
            path=path, file=csv_bytes, file_options={"content-type": "text/csv"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload processed dataset failed: {str(e)}"
        )

    return {"dataset_id": dataset_id, "columns": columns, "original_file_type": original_file_type}


def get_summary(dataset_id: str, owner_id: str, user: dict, target_column: str = None):
    target_user_id = user["user_id"]

    if user["role"] == "admin" and owner_id:
        target_user_id = owner_id

    df = load_dataset(dataset_id, target_user_id)

    if target_column and target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' not found. Available columns: {list(df.columns)}",
        )

    rows, cols = df.shape

    features_to_process = [col for col in df.columns if col != target_column]
    categorical_features = []
    numerical_features = []

    for column in features_to_process:
        if df[column].nunique() > 6:
            numerical_features.append(column)
        else:
            categorical_features.append(column)

    # ===== KIỂU DỮ LIỆU =====
    column_types = df.dtypes.astype(str).to_dict()

    # ===== GIÁ TRỊ THIẾU =====
    missing_values = df.isnull().sum().to_dict()

    return _sanitize(
        {
            "rows": rows,
            "columns": cols,
            "target_column": target_column,
            "categorical_features": categorical_features,
            "numerical_features": numerical_features,
            "column_types": column_types,
            "missing_values": missing_values,
        }
    )


def get_rows(dataset_id: str, owner_id: str, user: dict, limit: int, offset: int):
    target_user_id = user["user_id"]

    if user["role"] == "admin" and owner_id:
        target_user_id = owner_id

    df = load_dataset(dataset_id, target_user_id)

    total_rows = len(df)

    # 2. Thực hiện cắt dữ liệu (Slicing) theo limit và offset
    # Công thức: [vị trí bắt đầu : vị trí kết thúc]
    df_slice = df.iloc[offset : offset + limit]

    # 3. Xử lý giá trị NaN
    df_slice = df_slice.astype(object).where(pd.notnull(df_slice), None)

    # 4. Chuyển đổi sang list các dictionary
    data = df_slice.to_dict(orient="records")

    return {"total_rows": total_rows, "limit": limit, "offset": offset, "data": data}


def preprocess(dataset_id: str, owner_id: str, user: dict, target_column, imputation_method: str = "standard"):
    target_user_id = user["user_id"]

    if user["role"] == "admin" and owner_id:
        target_user_id = owner_id

    # 1. Load dữ liệu
    df = load_dataset(dataset_id, target_user_id)

    if target_column and target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' not found. Available columns: {list(df.columns)}",
        )

    if target_column and target_column in df.columns:
        df = df.dropna(subset=[target_column])

    # 2. Xử lý giá trị trùng lặp (Duplicates)
    rows_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before - len(df)

    features_to_process = [col for col in df.columns if col != target_column]

    # Phân loại cột
    categorical_features = []
    numerical_features = []
    unprocessed_columns = []

    for col in features_to_process:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        unique_count = df[col].nunique()

        if is_numeric and unique_count > 6:
            numerical_features.append(col)
        elif not is_numeric and unique_count > 15:
            unprocessed_columns.append(col)
        else:
            categorical_features.append(col)

    if unprocessed_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": f"Some columns cannot be preprocessed: {', '.join(unprocessed_columns)}"},
        )

    le = LabelEncoder()  # Khởi tạo LabelEncoder dùng chung cho cả 2 phương pháp

    if imputation_method == "mice":
        # Lấy tỷ lệ thiếu để áp dụng đúng phương pháp cho từng cột
        missing_percentages = df[features_to_process].isnull().mean() * 100
        cols_missing_under_5 = missing_percentages[
            (missing_percentages > 0) & (missing_percentages <= 5)
        ].index.tolist()
        cols_missing_over_5 = missing_percentages[missing_percentages > 5].index.tolist()

        # 3.1 Xử lý các cột thiếu <= 5% (Mean cho Numeric, Mode cho Categorical)
        for col in cols_missing_under_5:
            if col in numerical_features:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        # 3.2 Label Encode cho non-null values để thuật toán MICE có thể chạy được
        for col in categorical_features:
            non_nulls = df[col].dropna()
            if not non_nulls.empty:
                df.loc[df[col].notnull(), col] = le.fit_transform(non_nulls.astype(str))

        # 3.3 Áp dụng MICE cho các biến thiếu > 5%
        if len(cols_missing_over_5) > 0:
            mice_imputer = IterativeImputer(max_iter=10, random_state=0)
            df[features_to_process] = mice_imputer.fit_transform(df[features_to_process])

            # MICE có thể tạo số thập phân cho biến phân loại, nên cần làm tròn
            for col in categorical_features:
                df[col] = df[col].round()

    elif imputation_method == "mean":
        # Áp dụng Average Estimated Method: Dùng trung bình (Mean) cho số, Mode cho phân loại
        for col in features_to_process:
            if df[col].isnull().any():
                if col in numerical_features:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # Label Encoding cho các cột Categorical
        for col in categorical_features:
            df[col] = le.fit_transform(df[col].astype(str))

    elif imputation_method == "knn":
        # Áp dụng K-Nearest Neighbors với k=2 theo như kết quả bài báo[cite: 5]

        # Cần Encode tạm các giá trị Non-null sang dạng số để tính toán khoảng cách KNN[cite: 5]
        for col in categorical_features:
            non_nulls = df[col].dropna()
            if not non_nulls.empty:
                df.loc[df[col].notnull(), col] = le.fit_transform(non_nulls.astype(str))

        # Thực hiện nội suy bằng KNN với K=2
        knn_imputer = KNNImputer(n_neighbors=2)
        df[features_to_process] = knn_imputer.fit_transform(df[features_to_process])

        # Làm tròn kết quả đối với các biến phân loại sau khi KNN trả về số thập phân
        for col in categorical_features:
            df[col] = df[col].round()

    else:
        # XỬ LÝ MẶC ĐỊNH (STANDARD) BAN ĐẦu
        # 3. Xử lý giá trị thiếu (Missing Values)
        for col in features_to_process:
            if df[col].isnull().any():
                if col in numerical_features:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # 4. Label Encoding cho các cột Categorical
        for col in categorical_features:
            df[col] = le.fit_transform(df[col].astype(str))

    # 5. Scaling cho các cột Numerical
    scaler = StandardScaler()
    if numerical_features:
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # 7. Nếu Target là dạng chuỗi, chỉ Encode sang số
    if target_column and (df[target_column].dtype == "object" or df[target_column].dtype == "bool"):
        df[target_column] = le.fit_transform(df[target_column].astype(str))

    # 6. Lưu kết quả đã xử lý lên Supabase (vào thư mục 'processed/')
    processed_dataset_id = f"processed_{dataset_id}"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    path = f"processed/{target_user_id}/{processed_dataset_id}.csv"

    try:
        supabase.storage.from_(DATASET_BUCKET).upload(
            path=path, file=csv_bytes, file_options={"content-type": "text/csv", "upsert": "true"}
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f": {str(e)}")

    return {
        "message": "Preprocessing completed successfully",
        "original_dataset_id": dataset_id,
        "processed_dataset_id": processed_dataset_id,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicates_removed": duplicates_removed,
    }


def download(dataset_id: str, owner_id: str, user: dict, file_type: str):
    target_user_id = user["user_id"]

    if user["role"] == "admin" and owner_id:
        target_user_id = owner_id

    # Default to csv
    if not file_type:
        file_type = "csv"

    df = load_dataset(dataset_id, target_user_id)

    # Convert format
    if file_type.lower() in ["xlsx", "xls"]:
        stream = io.BytesIO()
        df.to_excel(stream, index=False, engine="openpyxl")
        stream.seek(0)

        response = StreamingResponse(
            iter([stream.getvalue()]), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={dataset_id}.xlsx"
    else:
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename={dataset_id}.csv"

    return response


def get_eda(dataset_id: str, target_column: str, owner_id: str, user: dict):
    target_user_id = owner_id if (user["role"] == "admin" and owner_id) else user["user_id"]
    df = load_dataset(dataset_id, target_user_id)

    if target_column and target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' not found. Available columns: {list(df.columns)}",
        )

    # Phân loại features
    features_only = [col for col in df.columns if col != target_column]
    categorical_features = [col for col in features_only if df[col].nunique() <= 6]
    numerical_features = [col for col in features_only if df[col].nunique() > 6]
    numeric_df = df.select_dtypes(include=["number"])

    stats = df.describe().to_dict()
    charts = {}

    # --- 1. FULL CORRELATION HEATMAP (Tương quan giữa các cặp biến) ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Full Correlation Heatmap")
    charts["full_correlation"] = upload_plot(plt, f"{target_user_id}/{dataset_id}/full_corr.png")

    if target_column and target_column in df.columns:
        target = df[target_column]

        # --- 2. TARGET DISTRIBUTION (Tỷ lệ và số lượng nhãn) ---
        counts = target.value_counts()
        palette = sns.color_palette("coolwarm", n_colors=len(counts))
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=palette,
            wedgeprops={"edgecolor": "black"},
        )
        ax[0].set_title(f"{target_column} %")
        sns.barplot(
            x=counts.index,
            y=counts.values,
            hue=counts.index,
            ax=ax[1],
            palette=palette,
            edgecolor="black",
            legend=False,
        )
        ax[1].set_title(f"Cases of {target_column}")
        ax[1].set_xlabel(target_column)
        ax[1].set_ylabel("count")
        charts["target_distribution"] = upload_plot(plt, f"{target_user_id}/{dataset_id}/dist.png")

        # --- 3. TARGET CORRELATION (Pearson - Tương quan tuyến tính với nhãn) ---
        if target_column in numeric_df.columns:
            target_corr = numeric_df.corrwith(numeric_df[target_column]).sort_values(ascending=False).to_frame()
            target_corr.columns = ["Correlation"]
            plt.figure(figsize=(6, 10))
            sns.heatmap(target_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title(f"Correlation w.r.t {target_column}")
            charts["target_correlation"] = upload_plot(plt, f"{target_user_id}/{dataset_id}/target_corr.png")

        # --- 4. CHI-SQUARE SCORE (Ý nghĩa của biến phân loại) ---
        if categorical_features:
            X_cat = df[categorical_features].fillna(0).abs()
            fit_cat = SelectKBest(score_func=chi2, k="all").fit(X_cat, target)
            cat_scores = pd.DataFrame(data=fit_cat.scores_, index=categorical_features, columns=["Score"]).sort_values(
                by="Score", ascending=False
            )
            plt.figure(figsize=(6, 8))
            sns.heatmap(cat_scores, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Categorical Importance (Chi-Square)")
            charts["chi_square_score"] = upload_plot(plt, f"{target_user_id}/{dataset_id}/chi2.png")

        # --- 5. ANOVA SCORE (Ý nghĩa của biến số) ---
        if numerical_features:
            X_num = df[numerical_features].fillna(df[numerical_features].median())
            fit_num = SelectKBest(score_func=f_classif, k="all").fit(X_num, target)
            num_scores = pd.DataFrame(data=fit_num.scores_, index=numerical_features, columns=["Score"]).sort_values(
                by="Score", ascending=False
            )
            plt.figure(figsize=(6, 8))
            sns.heatmap(num_scores, annot=True, cmap="YlOrRd", fmt=".2f")
            plt.title("Numerical Importance (ANOVA)")
            charts["anova_score"] = upload_plot(plt, f"{target_user_id}/{dataset_id}/anova.png")

    return _sanitize({"statistics": stats, "charts": charts})


def initialization_of_population(size, n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=bool)
        chromosome[: int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population, model, X_train, X_test, Y_train, Y_test):
    scores = []
    for chromosome in population:
        X_train_selected = X_train.iloc[:, chromosome]
        X_test_selected = X_test.iloc[:, chromosome]
        model.fit(X_train_selected, Y_train)
        predictions = model.predict(X_test_selected)
        scores.append(accuracy_score(Y_test, predictions))

    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds, :][::-1])


def selection(pop_after_fit, n_parents):
    return list(pop_after_fit[:n_parents])


def crossover(pop_after_sel):
    pop_nextgen = list(pop_after_sel)
    for i in range(0, len(pop_after_sel) - 1, 2):
        child_1, child_2 = pop_nextgen[i], pop_nextgen[i + 1]
        new_par = np.concatenate((child_1[: len(child_1) // 2], child_2[len(child_1) // 2 :]))
        pop_nextgen.append(new_par)
    return pop_nextgen


def mutation(pop_after_cross, mutation_rate, n_feat):
    mutation_range = int(mutation_rate * n_feat)
    pop_next_gen = []
    for chromo in pop_after_cross:
        new_chromo = chromo.copy()
        for _ in range(mutation_range):
            pos = randint(0, n_feat - 1)
            new_chromo[pos] = not new_chromo[pos]
        pop_next_gen.append(new_chromo)
    return pop_next_gen


def generations(model, size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, Y_train, Y_test):
    best_chromo = []
    best_score = []
    population_nextgen = initialization_of_population(size, n_feat)

    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen, model, X_train, X_test, Y_train, Y_test)
        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)

        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])

    return best_chromo, best_score


def genetic_selection(
    dataset_id: str,
    target_column: str,
    owner_id: str,
    user: dict,
    size: int,
    n_gen: int,
    mutation_rate: float,
    n_parents: int,
    model_name: str,
    test_size: float,
    balancing_method: str = "none",
):
    if n_parents is None or n_parents >= size:
        n_parents = int(size * 0.8)

    target_user_id = owner_id if (user["role"] == "admin" and owner_id) else user["user_id"]
    processed_df = load_dataset(dataset_id, target_user_id)

    if target_column and target_column not in processed_df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' not found. Available columns: {list(processed_df.columns)}",
        )

    target = processed_df[target_column]
    features = processed_df.drop(columns=[target_column])

    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=test_size, random_state=42)

    # --- DATA BALANCING ---
    balancing_info = None
    if balancing_method == "smote":
        class_counts = Y_train.value_counts()
        if len(class_counts) >= 2 and class_counts.min() / class_counts.max() < 0.9:
            try:
                before_counts = Y_train.value_counts().to_dict()
                smote = SMOTE(random_state=42)
                X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
                categorical_cols = [col for col in X_train.columns if X_train[col].nunique() <= 6]
                for col in categorical_cols:
                    X_resampled[col] = X_resampled[col].round()

                X_train = X_resampled
                Y_train = pd.Series(Y_resampled, name=Y_train.name)
                after_counts = Y_train.value_counts().to_dict()
                balancing_info = {"method": "smote", "before": before_counts, "after": after_counts}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"SMOTE failed: {str(e)}",
                )
        else:
            balancing_info = {
                "method": "smote",
                "skipped": "Dataset is already balanced (ratio >= 0.9), SMOTE was not applied.",
            }
    elif balancing_method == "adasyn":
        class_counts = Y_train.value_counts()
        if len(class_counts) >= 2 and class_counts.min() / class_counts.max() < 0.9:
            try:
                before_counts = Y_train.value_counts().to_dict()
                adasyn = ADASYN(random_state=42)
                X_resampled, Y_resampled = adasyn.fit_resample(X_train, Y_train)

                categorical_cols = [col for col in X_train.columns if X_train[col].nunique() <= 6]
                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
                for col in categorical_cols:
                    X_resampled[col] = X_resampled[col].round()

                X_train = X_resampled
                Y_train = pd.Series(Y_resampled, name=Y_train.name)
                after_counts = Y_train.value_counts().to_dict()
                balancing_info = {"method": "adasyn", "before": before_counts, "after": after_counts}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"ADASYN failed: {str(e)}",
                )
        else:
            balancing_info = {
                "method": "adasyn",
                "skipped": "Dataset is already balanced (ratio >= 0.9), ADASYN was not applied.",
            }

    # --- CHỌN MÔ HÌNH ---
    default_model = SELECTED_MODEL

    if default_model and default_model not in AVAILABLE_MODELS:
        print(f"Model '{default_model}' is not supported. Available models: {list(AVAILABLE_MODELS.keys())}")
        default_model = "decision_tree"

    final_model = default_model

    if user["role"] == "admin" and model_name:
        if model_name in AVAILABLE_MODELS:
            final_model = model_name
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model_name}' is not supported. Available models: {list(AVAILABLE_MODELS.keys())}",
            )

    print(f"Using model {final_model} for feature selection")

    # Khởi tạo Model
    model = AVAILABLE_MODELS[final_model]

    # --- ĐÁNH GIÁ MÔ HÌNH BAN ĐẦU (BASELINE) ---
    model.fit(X_train, Y_train)
    baseline_predictions = model.predict(X_test)
    baseline_accuracy = accuracy_score(Y_test, baseline_predictions)
    original_feature_count = features.shape[1]
    # ------------------------------------------

    # Genetic Algorithm
    best_chromo_list, best_score_list = generations(
        model=model,
        size=size,
        n_feat=features.shape[1],
        n_parents=n_parents,
        mutation_rate=mutation_rate,
        n_gen=n_gen,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
    )

    # Tìm index của giá trị accuracy cao nhất trong các thế hệ
    best_gen_index = np.argmax(best_score_list)

    # Trích xuất score và chromosome tại thế hệ đó
    absolute_best_score = float(best_score_list[best_gen_index])
    absolute_best_chromo = best_chromo_list[best_gen_index]

    # Lấy danh sách feature
    selected_features = features.columns[absolute_best_chromo].tolist()

    columns_to_keep = selected_features.copy()
    if target_column and target_column not in columns_to_keep:
        columns_to_keep.append(target_column)

    df_selected = processed_df[columns_to_keep]

    fs_dataset_id = f"fs_{dataset_id}"
    csv_bytes = df_selected.to_csv(index=False).encode("utf-8")

    path = f"feature-selection/{target_user_id}/{fs_dataset_id}.csv"

    try:
        supabase.storage.from_(DATASET_BUCKET).upload(
            path=path, file=csv_bytes, file_options={"content-type": "text/csv", "upsert": "true"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload feature selection dataset failed: {str(e)}",
        )

    return {
        "fs_dataset_id": fs_dataset_id,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "original_feature_count": original_feature_count,
        "best_ga_accuracy": round(absolute_best_score, 4),
        "selected_features": selected_features,
        "feature_count": len(selected_features),
        "found_at_generation": int(best_gen_index + 1),
        "balancing": balancing_info,
    }


def evaluate_feature_selection(
    dataset_id: str,
    fs_dataset_id: str,
    target_column: str,
    owner_id: str,
    user: dict,
    model_name: str,
    test_size: float,
):
    target_user_id = owner_id if (user["role"] == "admin" and owner_id) else user["user_id"]

    # 2 tập dữ liệu từ Supabase Storage
    df_original = load_dataset(dataset_id, target_user_id)
    df_selected = load_dataset(fs_dataset_id, target_user_id)

    if target_column not in df_original.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    # Tách Features và Target
    Y_orig = df_original[target_column]
    X_orig = df_original.drop(columns=[target_column])

    Y_sel = df_selected[target_column]
    X_sel = df_selected.drop(columns=[target_column])

    # Chia tập Train/Test
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(
        X_orig, Y_orig, test_size=test_size, random_state=42
    )
    X_train_sel, X_test_sel, Y_train_sel, Y_test_sel = train_test_split(
        X_sel, Y_sel, test_size=test_size, random_state=42
    )

    # Xác định mô hình sử dụng
    final_model_name = model_name if (user["role"] == "admin" and model_name) else SELECTED_MODEL
    if final_model_name not in AVAILABLE_MODELS:
        final_model_name = "decision_tree"

    # LUÔN GIỮ COPY ĐỂ BẢO VỆ SERVER
    model_before = copy.deepcopy(AVAILABLE_MODELS[final_model_name])
    model_after = copy.deepcopy(AVAILABLE_MODELS[final_model_name])

    def _to_frame(data, columns):
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data, columns=columns)

    def _predict_with_columns(model, columns):
        def _predict(data):
            return model.predict(_to_frame(data, columns))

        return _predict

    def _predict_proba_with_columns(model, columns):
        def _predict_proba(data):
            return model.predict_proba(_to_frame(data, columns))

        return _predict_proba

    # =================================================================
    # ĐÁNH GIÁ MÔ HÌNH TRƯỚC (BASELINE)
    # =================================================================
    model_before.fit(X_train_orig, Y_train_orig)
    preds_before = model_before.predict(X_test_orig)

    metrics_before = {
        "accuracy": round(accuracy_score(Y_test_orig, preds_before), 4),
        "precision": round(precision_score(Y_test_orig, preds_before, average="binary", zero_division=0), 4),
        "recall": round(recall_score(Y_test_orig, preds_before, average="binary", zero_division=0), 4),
        "f1_score": round(f1_score(Y_test_orig, preds_before, average="binary", zero_division=0), 4),
    }
    cm_before = confusion_matrix(Y_test_orig, preds_before)

    # =================================================================
    # ĐÁNH GIÁ MÔ HÌNH SAU GA (OPTIMIZED)
    # =================================================================
    model_after.fit(X_train_sel, Y_train_sel)
    preds_after = model_after.predict(X_test_sel)

    metrics_after = {
        "accuracy": round(accuracy_score(Y_test_sel, preds_after), 4),
        "precision": round(precision_score(Y_test_sel, preds_after, average="binary", zero_division=0), 4),
        "recall": round(recall_score(Y_test_sel, preds_after, average="binary", zero_division=0), 4),
        "f1_score": round(f1_score(Y_test_sel, preds_after, average="binary", zero_division=0), 4),
    }
    cm_after = confusion_matrix(Y_test_sel, preds_after)

    # =================================================================
    # VẼ ĐỒ THỊ CONFUSION MATRIX
    # =================================================================
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.heatmap(
        cm_before,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes_cm[0],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes_cm[0].set_title("Confusion Matrix (Before Selection)", fontsize=12, pad=10)
    axes_cm[0].set_xlabel("Predicted Label")
    axes_cm[0].set_ylabel("Actual Label")

    sns.heatmap(
        cm_after,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes_cm[1],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    axes_cm[1].set_title("Confusion Matrix (After Selection)", fontsize=12, pad=10)
    axes_cm[1].set_xlabel("Predicted Label")
    axes_cm[1].set_ylabel("Actual Label")

    plt.tight_layout()
    chart_url = upload_plot(plt, f"{target_user_id}/{dataset_id}/confusion_matrix_comparison.png")

    # =================================================================
    # VẼ BIỂU ĐỒ ROC CURVE SO SÁNH
    # =================================================================
    roc_chart_url = None
    try:
        from sklearn.metrics import roc_curve, auc

        # Lấy xác suất dự đoán (class 1)
        probs_before = model_before.predict_proba(X_test_orig)[:, 1]
        probs_after = model_after.predict_proba(X_test_sel)[:, 1]

        # Tính ROC curve
        fpr_before, tpr_before, _ = roc_curve(Y_test_orig, probs_before)
        roc_auc_before = auc(fpr_before, tpr_before)

        fpr_after, tpr_after, _ = roc_curve(Y_test_sel, probs_after)
        roc_auc_after = auc(fpr_after, tpr_after)

        # Vẽ đồ thị
        fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
        ax_roc.plot(
            fpr_before,
            tpr_before,
            color="darkorange",
            lw=2,
            linestyle="--",
            label=f"Before Selection (AUC = {roc_auc_before:.3f})",
        )
        ax_roc.plot(fpr_after, tpr_after, color="green", lw=2.5, label=f"After Selection (AUC = {roc_auc_after:.3f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax_roc.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, pad=15)
        ax_roc.legend(loc="lower right", frameon=True, shadow=True)

        plt.tight_layout()
        roc_chart_url = upload_plot(plt, f"{target_user_id}/{dataset_id}/roc_comparison.png")
    except Exception as e:
        print(f"Error gen ROC plot: {e}")

    # =================================================================
    # VẼ BIỂU ĐỒ SHAP (BAR & BEESWARM) TRƯỚC VÀ SAU
    # =================================================================
    shap_chart_before_url = None
    shap_chart_after_url = None
    shap_beeswarm_before_url = None
    shap_beeswarm_after_url = None

    try:
        limit = 25
        X_sample_orig = shap.utils.sample(X_test_orig, limit) if len(X_test_orig) > limit else X_test_orig
        X_sample_sel = shap.utils.sample(X_test_sel, limit) if len(X_test_sel) > limit else X_test_sel

        tree_models = ["xgboost", "lightgbm", "random_forest", "decision_tree"]

        def _compute_shap_values(model, X_sample, X_train_full):
            if final_model_name in tree_models:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_sample)
            elif final_model_name == "logistic_regression":
                explainer = shap.LinearExplainer(model, X_train_full)
                sv = explainer.shap_values(X_sample)
            else:
                background = shap.kmeans(X_train_full, 2)
                explainer = shap.KernelExplainer(
                    _predict_with_columns(model, X_train_full.columns),
                    background,
                )
                sv = explainer.shap_values(X_sample, silent=True)

            if isinstance(sv, list):
                sv = sv[1]
            elif len(sv.shape) == 3:
                sv = sv[:, :, 1]
            return sv

        # --- 1. TÍNH TOÁN & VẼ SHAP BEFORE ---
        shap_values_before = _compute_shap_values(model_before, X_sample_orig, X_train_orig)

        # Bar Chart Before
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values_before, X_sample_orig, plot_type="bar", show=False)
        plt.title("SHAP Global Feature Importance (Before Feature Selection)", pad=15, fontsize=12)
        plt.tight_layout()
        shap_chart_before_url = upload_plot(plt, f"{target_user_id}/{dataset_id}/shap_before.png")

        # Beeswarm Chart Before (Tận dụng luôn shap_values_before đã tính)
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values_before, X_sample_orig, show=False)  # Mặc định là dot/beeswarm
        plt.title("SHAP Beeswarm Distribution (Before Feature Selection)", pad=15, fontsize=12)
        plt.tight_layout()
        shap_beeswarm_before_url = upload_plot(plt, f"{target_user_id}/{dataset_id}/shap_beeswarm_before.png")

        # --- 2. TÍNH TOÁN & VẼ SHAP AFTER ---
        shap_values_after = _compute_shap_values(model_after, X_sample_sel, X_train_sel)

        # Bar Chart After
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values_after, X_sample_sel, plot_type="bar", show=False)
        plt.title("SHAP Global Feature Importance (After Feature Selection)", pad=15, fontsize=12)
        plt.tight_layout()
        shap_chart_after_url = upload_plot(plt, f"{target_user_id}/{dataset_id}/shap_after.png")

        # Beeswarm Chart After
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values_after, X_sample_sel, show=False)
        plt.title("SHAP Beeswarm Distribution (After Feature Selection)", pad=15, fontsize=12)
        plt.tight_layout()
        shap_beeswarm_after_url = upload_plot(plt, f"{target_user_id}/{dataset_id}/shap_beeswarm_after.png")

    except Exception as e:
        print(f"Error generating SHAP plots: {e}")
        plt.close("all")

    # =================================================================
    # VẼ BIỂU ĐỒ LIME LOCAL EXPLANATION
    # =================================================================
    lime_chart_before_url = None
    lime_chart_after_url = None

    try:
        from lime import lime_tabular

        # Chọn dòng đầu tiên của tập test để giải thích
        instance_idx = 0

        # LIME Before
        explainer_before = lime_tabular.LimeTabularExplainer(
            training_data=X_train_orig.values,
            feature_names=X_train_orig.columns.tolist(),
            class_names=["Class 0", "Class 1"],
            mode="classification",
            random_state=42,
        )

        exp_before = explainer_before.explain_instance(
            data_row=X_test_orig.iloc[instance_idx].values,
            predict_fn=_predict_proba_with_columns(model_before, X_train_orig.columns),
            num_features=10,  # Top 10 feature quan trọng nhất cho mẫu này
        )

        # as_pyplot_figure() tự động tạo một matplotlib figure
        fig_before = exp_before.as_pyplot_figure()
        plt.title(f"LIME Local Explanation - Instance {instance_idx} (Before Feature Selection)", pad=15)
        plt.tight_layout()
        lime_chart_before_url = upload_plot(plt, f"{target_user_id}/{dataset_id}/lime_before.png")

        # LIME After
        explainer_after = lime_tabular.LimeTabularExplainer(
            training_data=X_train_sel.values,
            feature_names=X_train_sel.columns.tolist(),
            class_names=["Class 0", "Class 1"],
            mode="classification",
            random_state=42,
        )

        exp_after = explainer_after.explain_instance(
            data_row=X_test_sel.iloc[instance_idx].values,
            predict_fn=_predict_proba_with_columns(model_after, X_train_sel.columns),
            num_features=10,
        )

        fig_after = exp_after.as_pyplot_figure()
        plt.title(f"LIME Local Explanation - Instance {instance_idx} (After Feature Selection)", pad=15)
        plt.tight_layout()
        lime_chart_after_url = upload_plot(plt, f"{target_user_id}/{dataset_id}/lime_after.png")

    except Exception as e:
        print(f"Error generating LIME plots: {e}")
        plt.close("all")

    return _sanitize(
        {
            "status": "success",
            "model_evaluated": final_model_name,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "confusion_matrix_chart_url": chart_url,
            "roc_chart_url": roc_chart_url,
            "shap_chart_before_url": shap_chart_before_url,
            "shap_chart_after_url": shap_chart_after_url,
            "shap_beeswarm_before_url": shap_beeswarm_before_url,
            "shap_beeswarm_after_url": shap_beeswarm_after_url,
            "lime_chart_before_url": lime_chart_before_url,
            "lime_chart_after_url": lime_chart_after_url,
        }
    )
