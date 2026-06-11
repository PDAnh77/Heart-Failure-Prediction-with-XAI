import os
import time
import warnings
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from datetime import datetime
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# Create directory for saving images
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
image_dir = os.path.join("result_images", current_time)
os.makedirs(image_dir, exist_ok=True)
print(f"Created directory for saving images: {image_dir}")

TEST_SIZE = 0.3
LOCAL_INSTANCE_IDX = 0 # Choose a local instance for specific explanations (e.g., the first test record)

classifiers = [
    "SVC",
    "LogisticRegression",
    "RandomForest",
    "DecisionTree",
    "KNeighbors",
    "XGBoost",
    "LightGBM",
]

models = [
    SVC(kernel="linear", C=0.1, probability=True),
    LogisticRegression(random_state=0, C=10, penalty="l2"),
    RandomForestClassifier(max_depth=4, random_state=0),
    DecisionTreeClassifier(random_state=1000, max_depth=4, min_samples_leaf=1),
    KNeighborsClassifier(leaf_size=1, n_neighbors=3, p=1),
    XGBClassifier(
        random_state=0,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.105,
        subsample=0.8,
        colsample_bytree=0.9,
        eval_metric="logloss",
    ),
    LGBMClassifier(
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
]


def split(features, target):
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=TEST_SIZE, random_state=42)
    return X_train, X_test, Y_train, Y_test


def acc_score(df, label):
    Score = pd.DataFrame({"Classifier": classifiers})
    acc, prec, rec, f1 = [], [], [], []
    X_train, X_test, Y_train, Y_test = split(df, label)
    
    for model in models:
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        
        acc.append(accuracy_score(Y_test, predictions))
        prec.append(precision_score(Y_test, predictions, zero_division=0))
        rec.append(recall_score(Y_test, predictions, zero_division=0))
        f1.append(f1_score(Y_test, predictions, zero_division=0))
        
    Score["Accuracy"] = acc
    Score["Precision"] = prec
    Score["Recall"] = rec
    Score["F1_Score"] = f1
    Score.sort_values(by="Accuracy", ascending=False, inplace=True)
    Score.reset_index(drop=True, inplace=True)
    return Score


def plot(score, x, y, c="b"):
    gen = list(range(1, len(score) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(gen, score, marker="o", color=c)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.ylim(x, y)
    plt.grid(True)


def save_df_as_image(df, filepath, title):
    df_copy = df.copy()
    has_features = "Selected_Features" in df_copy.columns

    if has_features:
        df_copy["Selected_Features"] = df_copy["Selected_Features"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )

    df_copy = df_copy.round(4)

    if has_features:
        top_cols = [c for c in df_copy.columns if c != "Selected_Features"]
        cell_data = []
        for index, row in df_copy.iterrows():
            cell_data.append([str(row[c]) for c in top_cols])
            cell_data.append(["Selected_Features"] + [""] * (len(top_cols) - 1))
            cell_data.append([str(row["Selected_Features"])] + [""] * (len(top_cols) - 1))
        headers = top_cols
    else:
        cell_data = df_copy.values.tolist()
        headers = df_copy.columns.tolist()

    fig_height = 0.4 * len(cell_data) + 1.5
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    table = ax.table(cellText=cell_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    if has_features:
        num_cols = len(top_cols)
        for i in range(1, len(cell_data) + 1):
            if i % 3 == 2:
                for j in range(num_cols):
                    cell = table[i, j]
                    if j == 0:
                        cell.visible_edges = "LT"
                        cell.set_text_props(weight="bold", ha="left")
                        cell.get_text().set_text("   " + cell.get_text().get_text())
                    elif j == num_cols - 1:
                        cell.visible_edges = "RT"
                    else:
                        cell.visible_edges = "T"

            elif i % 3 == 0:
                for j in range(num_cols):
                    cell = table[i, j]
                    if j == 0:
                        cell.visible_edges = "LB"
                        cell.set_text_props(ha="left")
                        cell.get_text().set_text("   " + cell.get_text().get_text())
                    elif j == num_cols - 1:
                        cell.visible_edges = "RB"
                    else:
                        cell.visible_edges = "B"

    plt.title(title, fontweight="bold", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"Saved table image: {filepath}")
    plt.close()


def initilization_of_population(size, n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool_)
        chromosome[: int(0.3 * n_feat)] = False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population, model, X_train, X_test, Y_train, Y_test):
    scores = []
    for chromosome in population:
        if np.sum(chromosome) == 0:
            scores.append(0.0) 
            continue

        X_train_selected = X_train.iloc[:, chromosome]
        X_test_selected = X_test.iloc[:, chromosome]

        model.fit(X_train_selected, Y_train)
        predictions = model.predict(X_test_selected)
        scores.append(accuracy_score(Y_test, predictions))

    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds, :][::-1])


def selection(pop_after_fit, n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0, len(pop_after_sel), 2):
        new_par = []
        child_1, child_2 = pop_nextgen[i], pop_nextgen[i + 1]
        new_par = np.concatenate((child_1[: len(child_1) // 2], child_2[len(child_1) // 2 :]))
        pop_nextgen.append(new_par)
    return pop_nextgen


def mutation(pop_after_cross, mutation_rate, n_feat):
    mutation_range = int(mutation_rate * n_feat)
    pop_next_gen = []
    for n in range(0, len(pop_after_cross)):
        chromo = pop_after_cross[n].copy()
        rand_posi = []
        for i in range(0, mutation_range):
            pos = randint(0, n_feat - 1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop_next_gen.append(chromo)
    return pop_next_gen


def generations(model, df, label, size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, Y_train, Y_test):
    best_chromo = []
    best_score = []
    population_nextgen = initilization_of_population(size, n_feat)

    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen, model, X_train, X_test, Y_train, Y_test)
        print("Best score in generation", i + 1, ":", scores[:1])
        pop_after_sel = selection(pop_after_fit, n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)

        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])

    return best_chromo, best_score

# ==========================================
# EXPLAINABILITY AND COMPARISON FUNCTIONS
# ==========================================

def generate_explainability_plots(model, X_train, X_test, model_name, state_prefix, instance_idx=0):
    print(f"\n--- Generating SHAP and LIME Explanations ({state_prefix}) ---")
    
    # 1. LIME Local Explanation
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['No Disease', 'Disease'],
        mode='classification'
    )
    
    predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
    exp = lime_explainer.explain_instance(
        data_row=X_test.iloc[instance_idx].values,
        predict_fn=predict_fn
    )
    
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Local Explanation - {model_name} ({state_prefix})", y=1.05)
    plt.tight_layout()
    lime_path = os.path.join(image_dir, f"LIME_{state_prefix}_{model_name}.png")
    plt.savefig(lime_path, bbox_inches="tight")
    print(f"Saved LIME plot: {lime_path}")
    plt.close()

    # 2. SHAP Explanations
    try:
        # Force probability output + disable additivity check for LightGBM/XGBoost
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model, X_train, model_output="probability")
            shap_values = explainer(X_test, check_additivity=False)
        else:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

    except Exception as e:
        print(f"Standard SHAP explainer failed, using KernelExplainer. Error: {e}")
        X_train_summary = shap.kmeans(X_train, 10)

        # Wrap in a plain function to avoid SHAP introspecting LightGBM's read-only properties
        def safe_predict(X):
            return model.predict_proba(X) if hasattr(model, 'predict_proba') else model.predict(X)

        explainer = shap.KernelExplainer(safe_predict, X_train_summary)
        shap_vals = explainer.shap_values(X_test)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif len(np.shape(shap_vals)) == 3:
            shap_vals = shap_vals[:, :, 1]

        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
        shap_values = shap.Explanation(
            values=shap_vals,
            base_values=np.full(X_test.shape[0], expected_value),
            data=X_test.values,
            feature_names=X_test.columns.tolist()
        )

    # Global Beeswarm
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(f"SHAP Global Beeswarm - {model_name} ({state_prefix})", fontsize=14)
    plt.tight_layout()
    path_beeswarm = os.path.join(image_dir, f"SHAP_Beeswarm_{state_prefix}_{model_name}.png")
    plt.savefig(path_beeswarm, bbox_inches="tight")
    print(f"Saved SHAP Beeswarm plot: {path_beeswarm}")
    plt.close()

    # Global Bar
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title(f"SHAP Global Bar Chart - {model_name} ({state_prefix})", fontsize=14)
    plt.tight_layout()
    path_global_bar = os.path.join(image_dir, f"SHAP_Global_Bar_{state_prefix}_{model_name}.png")
    plt.savefig(path_global_bar, bbox_inches="tight")
    print(f"Saved SHAP Global Bar plot: {path_global_bar}")
    plt.close()

    # Local Waterfall
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[instance_idx], show=False)
    plt.title(f"SHAP Local Waterfall (Instance {instance_idx}) - {model_name} ({state_prefix})", fontsize=14)
    plt.tight_layout()
    path_waterfall = os.path.join(image_dir, f"SHAP_Waterfall_{state_prefix}_{model_name}.png")
    plt.savefig(path_waterfall, bbox_inches="tight")
    print(f"Saved SHAP Waterfall plot: {path_waterfall}")
    plt.close()
    
    # Local Bar
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values[instance_idx], show=False)
    plt.title(f"SHAP Local Bar Chart (Instance {instance_idx}) - {model_name} ({state_prefix})", fontsize=14)
    plt.tight_layout()
    path_local_bar = os.path.join(image_dir, f"SHAP_Local_Bar_{state_prefix}_{model_name}.png")
    plt.savefig(path_local_bar, bbox_inches="tight")
    print(f"Saved SHAP Local Bar plot: {path_local_bar}")
    plt.close()


def plot_comparison_metrics(model_before, model_after, X_test_before, X_test_after, Y_test, model_name):
    print("\n--- Generating Confusion Matrix & ROC Comparisons ---")
    
    y_pred_b = model_before.predict(X_test_before)
    y_prob_b = model_before.predict_proba(X_test_before)[:, 1] if hasattr(model_before, "predict_proba") else y_pred_b

    y_pred_a = model_after.predict(X_test_after)
    y_prob_a = model_after.predict_proba(X_test_after)[:, 1] if hasattr(model_after, "predict_proba") else y_pred_a

    # 1. Confusion Matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cm_before = confusion_matrix(Y_test, y_pred_b)
    cm_after = confusion_matrix(Y_test, y_pred_a)

    ConfusionMatrixDisplay(cm_before).plot(ax=ax[0], cmap='Blues', colorbar=False)
    ax[0].set_title(f"Confusion Matrix (Before GA)\n{model_name}")

    ConfusionMatrixDisplay(cm_after).plot(ax=ax[1], cmap='Greens', colorbar=False)
    ax[1].set_title(f"Confusion Matrix (After GA)\n{model_name}")

    plt.tight_layout()
    cm_path = os.path.join(image_dir, f"Comparison_CM_{model_name}.png")
    plt.savefig(cm_path)
    print(f"Saved Confusion Matrix Comparison: {cm_path}")
    plt.close()

    # 2. ROC Curve
    fpr_b, tpr_b, _ = roc_curve(Y_test, y_prob_b)
    roc_auc_b = auc(fpr_b, tpr_b)

    fpr_a, tpr_a, _ = roc_curve(Y_test, y_prob_a)
    roc_auc_a = auc(fpr_a, tpr_a)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, color='blue', lw=2, label=f'Before GA (AUC = {roc_auc_b:.3f})')
    plt.plot(fpr_a, tpr_a, color='green', lw=2, label=f'After GA (AUC = {roc_auc_a:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    roc_path = os.path.join(image_dir, f"Comparison_ROC_{model_name}.png")
    plt.savefig(roc_path)
    print(f"Saved ROC Comparison: {roc_path}")
    plt.close()


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

data = pd.read_csv("../input/heart-failure-prediction/heart.csv")
df1 = data.copy(deep=True)

# Label Encoding
le_sex, le_chest, le_ecg, le_angina, le_slope = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()

df1["Sex"] = le_sex.fit_transform(df1["Sex"])
df1["ChestPainType"] = le_chest.fit_transform(df1["ChestPainType"])
df1["RestingECG"] = le_ecg.fit_transform(df1["RestingECG"])
df1["ExerciseAngina"] = le_angina.fit_transform(df1["ExerciseAngina"])
df1["ST_Slope"] = le_slope.fit_transform(df1["ST_Slope"])

# MinMaxScaler & StandardScaler
mms = MinMaxScaler()
df1["Oldpeak"] = mms.fit_transform(df1[["Oldpeak"]])

std_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR"]
ss = StandardScaler()
df1[std_cols] = ss.fit_transform(df1[std_cols])

target = df1["HeartDisease"]
features = df1[df1.columns.drop(["HeartDisease"])]

print("Heart Failure dataset:\n", features.shape[0], "Records\n", features.shape[1], "Features")
print("All the features in this dataset have continuous values")

# Calculate Score Before GA
score_before_ga = acc_score(features, target)
print(score_before_ga)

save_path_before_ga = os.path.join(image_dir, "Table_Before_GA.png")
save_df_as_image(score_before_ga, save_path_before_ga, "Model Accuracy Before GA")

X_train, X_test, Y_train, Y_test = split(features, target)

# ---------------------------------------------------------
# RUN GENETIC ALGORITHM
# ---------------------------------------------------------
print("\n--- Starting Feature Selection Optimization using Genetic Algorithm ---")
final_results = []

for name, model_obj in zip(classifiers, models):
    print(f"\nRunning GA for model: {name}...")
    start_time = time.perf_counter()

    best_chromo_list, best_score_list = generations(
        model=model_obj, df=features, label=target, size=80, n_feat=features.shape[1],
        n_parents=64, mutation_rate=0.20, n_gen=10,
        X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
    )

    elapsed_time = time.perf_counter() - start_time

    plot(best_score_list, 0.8, 1.0, c="orange")
    plt.title(f"GA Optimization Progress: {name}")
    save_path = os.path.join(image_dir, f"GA_Progress_{name}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    best_gen_index = np.argmax(best_score_list)
    best_chromosome = best_chromo_list[best_gen_index]
    overall_best_score = best_score_list[best_gen_index]
    overall_best_features = features.columns[best_chromosome].tolist()

    X_train_best = X_train.iloc[:, best_chromosome]
    X_test_best = X_test.iloc[:, best_chromosome]

    model_obj.fit(X_train_best, Y_train)
    predictions_best = model_obj.predict(X_test_best)

    precision = precision_score(Y_test, predictions_best)
    recall = recall_score(Y_test, predictions_best)
    f1 = f1_score(Y_test, predictions_best)

    final_results.append({
        "Classifier": name,
        "Accuracy": overall_best_score,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Selected_Features": overall_best_features,
        "Feature_Count": len(overall_best_features),
        "Best_Generation": best_gen_index + 1,
        "Execution_Time_Seconds": round(elapsed_time, 2),
    })

pd.set_option("display.max_colwidth", None)
df_final_comparison = pd.DataFrame(final_results).sort_values(by="Accuracy", ascending=False)

print("\n--- COMPARISON TABLE AFTER GA ---")
df_final_comparison.reset_index(drop=True, inplace=True)
print(df_final_comparison.to_string())

save_path_after_ga = os.path.join(image_dir, "Table_After_GA.png")
save_df_as_image(df_final_comparison, save_path_after_ga, "Model Comparison After GA Feature Selection")

# ---------------------------------------------------------
# VISUALIZATIONS FOR THE BEST PERFORMING MODEL
# ---------------------------------------------------------
# Step 1: Identify the absolute best algorithm from the post-GA results
best_row_after = df_final_comparison.iloc[0]
best_model_name = best_row_after["Classifier"]
best_features_list = best_row_after["Selected_Features"]

print(f"\nTop performing model AFTER Feature Selection: {best_model_name}")
print(f"Using features: {best_features_list}")

best_model_idx = classifiers.index(best_model_name)

# Step 2: Create a Baseline version of this best algorithm (trained on ALL features)
model_baseline = clone(models[best_model_idx])
model_baseline.fit(X_train, Y_train)

# Step 3: Create an Optimized version of this best algorithm (trained on SELECTED features)
model_optimized = clone(models[best_model_idx])
X_train_shap = X_train[best_features_list]
X_test_shap = X_test[best_features_list]
model_optimized.fit(X_train_shap, Y_train)

# Step 4: Generate explanations for the Baseline model
generate_explainability_plots(
    model=model_baseline, 
    X_train=X_train, 
    X_test=X_test, 
    model_name=best_model_name, 
    state_prefix="Before_GA", 
    instance_idx=LOCAL_INSTANCE_IDX
)

# Step 5: Generate explanations for the Optimized model
generate_explainability_plots(
    model=model_optimized, 
    X_train=X_train_shap, 
    X_test=X_test_shap, 
    model_name=best_model_name, 
    state_prefix="After_GA", 
    instance_idx=LOCAL_INSTANCE_IDX
)

# Step 6: Compare performance metrics (Confusion Matrix & ROC) directly
plot_comparison_metrics(
    model_before=model_baseline,
    model_after=model_optimized,
    X_test_before=X_test,
    X_test_after=X_test_shap,
    Y_test=Y_test,
    model_name=best_model_name
)