import io
from random import randint
import uuid
from datetime import datetime
from fastapi import HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier
from core.supabase_client import supabase

DATASET_BUCKET = "heart-failure-datasets"
EDA_BUCKET = "eda-artifacts"


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

    if "processed" in dataset_id:
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


def upload_dataset(file: UploadFile, user_id: str):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV files are allowed")

    # Check file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    # Read csv file
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid CSV file")

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
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    path = f"raw/{user_id}/{dataset_id}.csv"

    # Upload
    try:
        supabase.storage.from_(DATASET_BUCKET).upload(
            path=path, file=csv_bytes, file_options={"content-type": "text/csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload failed: {str(e)}")

    return {"dataset_id": dataset_id, "columns": columns}


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

    return _sanitize({
        "rows": rows,
        "columns": cols,
        "target_column": target_column,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "column_types": column_types,
        "missing_values": missing_values,
    })


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


def preprocess(dataset_id: str, owner_id: str, user: dict, target_column):
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

    # Phân loại cột dựa trên logic bạn đã dùng ở hàm summary
    categorical_features = [col for col in features_to_process if df[col].nunique() <= 6]
    numerical_features = [col for col in features_to_process if df[col].nunique() > 6]

    # 3. Xử lý giá trị thiếu (Missing Values)
    for col in features_to_process:
        if df[col].isnull().any():
            if col in numerical_features:
                # Điền giá trị thiếu bằng Trung vị (Median) cho cột số
                df[col] = df[col].fillna(df[col].median())
            else:
                # Điền bằng giá trị xuất hiện nhiều nhất (Mode) cho cột phân loại
                df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Label Encoding cho các cột Categorical
    # Chuyển đổi text/category thành số (0, 1, 2...)
    le = LabelEncoder()
    for col in categorical_features:
        df[col] = le.fit_transform(df[col].astype(str))

    # 5. Scaling cho các cột Numerical
    # Ở đây dùng StandardScaler. Nếu muốn dùng Min-Max, hãy thay bằng MinMaxScaler()
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Lỗi khi lưu dữ liệu đã xử lý: {str(e)}"
        )

    return {
        "message": "Preprocessing completed successfully",
        "original_dataset_id": dataset_id,
        "processed_dataset_id": processed_dataset_id,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicates_removed": duplicates_removed,
    }


def download(dataset_id: str, owner_id, user: dict):
    target_user_id = user["user_id"]

    if user["role"] == "admin" and owner_id:
        target_user_id = owner_id

    df = load_dataset(dataset_id, target_user_id)

    stream = io.StringIO()
    df.to_csv(stream, index=False)

    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=Processed dataset.csv"

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
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax[1], palette=palette, edgecolor="black", legend=False)
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

    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Khởi tạo Model
    model = XGBClassifier(
        random_state=0,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.105,
        subsample=0.8,
        colsample_bytree=0.9,
        eval_metric="logloss",
    )

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

    return {
        "baseline_accuracy": round(baseline_accuracy, 4),
        "original_feature_count": original_feature_count,
        "best_ga_accuracy": round(absolute_best_score, 4),
        "selected_features": selected_features,
        "feature_count": len(selected_features),
        "found_at_generation": int(best_gen_index + 1),
    }
