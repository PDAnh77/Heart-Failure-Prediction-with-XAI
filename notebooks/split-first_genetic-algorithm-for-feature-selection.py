import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

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
    SVC(kernel="linear", C=0.1),
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


def acc_score(X_train, X_test, Y_train, Y_test):
    Score = pd.DataFrame({"Classifier": classifiers})
    acc = []
    for i in models:
        model = i
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        acc.append(accuracy_score(Y_test, predictions))
    Score["Accuracy"] = acc
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


def initilization_of_population(size, n_feat):
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
        chromo = pop_after_cross[n]
        rand_posi = []
        for i in range(0, mutation_range):
            pos = randint(0, n_feat - 1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop_next_gen.append(chromo)
    return pop_next_gen


def generations(model, size, n_feat, n_parents, mutation_rate, n_gen, X_train, X_test, Y_train, Y_test):
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


# ----------------------------------------------------------
# Load dữ liệu
# ----------------------------------------------------------
data = pd.read_csv("../input/heart-failure-prediction/heart.csv")
df1 = data.copy(deep=True)

# ----------------------------------------------------------
# Tách target và features
# ----------------------------------------------------------
target = df1["HeartDisease"]
features = df1.drop(columns=["HeartDisease"])

print("Heart Failure dataset:\n", features.shape[0], "Records\n", features.shape[1], "Features")
print(features.head())

# ----------------------------------------------------------
# BƯỚC 1: CHIA TRAIN/TEST TRƯỚC — tránh Data Leakage
# ----------------------------------------------------------
print("--- Chia tập Train/Test trước khi tiền xử lý ---")

X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# BƯỚC 2: LABEL ENCODING (fit trên Train, transform trên Test)
# ----------------------------------------------------------
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=float)
X_train[categorical_cols] = oe.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = oe.transform(X_test[categorical_cols])

# ----------------------------------------------------------
# BƯỚC 3: SCALING (fit trên Train, transform trên Test)
# ----------------------------------------------------------
# MinMaxScaler cho Oldpeak
mms = MinMaxScaler()
X_train["Oldpeak"] = mms.fit_transform(X_train[["Oldpeak"]])
X_test["Oldpeak"] = mms.transform(X_test[["Oldpeak"]])

# StandardScaler cho numerical
std_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR"]
ss = StandardScaler()
X_train[std_cols] = ss.fit_transform(X_train[std_cols])
X_test[std_cols] = ss.transform(X_test[std_cols])

print("All the features in this dataset have continuous values")

score1 = acc_score(X_train, X_test, Y_train, Y_test)
print(score1)

# ----------------------------------------------------------
# BƯỚC 4: GENETIC ALGORITHM FEATURE SELECTION
# ----------------------------------------------------------
print("--- Starting Feature Selection Optimization using Genetic Algorithm ---")
final_results = []

for name, model_obj in zip(classifiers, models):
    print(f"\nRunning GA for model: {name}...")

    start_time = time.perf_counter()

    best_chromo_list, best_score_list = generations(
        model=model_obj,
        size=80,
        n_feat=X_train.shape[1],
        n_parents=64,
        mutation_rate=0.20,
        n_gen=10,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    plot(best_score_list, 0.8, 1.0, c="orange")
    plt.title(f"GA Optimization Progress: {name}")
    plt.show()

    best_gen_index = np.argmax(best_score_list)
    best_chromosome = best_chromo_list[best_gen_index]

    overall_best_score = best_score_list[best_gen_index]
    overall_best_features = X_train.columns[best_chromosome].tolist()

    X_train_best = X_train.iloc[:, best_chromosome]
    X_test_best = X_test.iloc[:, best_chromosome]

    model_obj.fit(X_train_best, Y_train)
    predictions_best = model_obj.predict(X_test_best)

    precision = precision_score(Y_test, predictions_best)
    recall = recall_score(Y_test, predictions_best)
    f1 = f1_score(Y_test, predictions_best)

    final_results.append(
        {
            "Classifier": name,
            "Accuracy": overall_best_score,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "Selected_Features": overall_best_features,
            "Feature_Count": len(overall_best_features),
            "Best_Generation": best_gen_index + 1,
            "Execution_Time_Seconds": round(elapsed_time, 2),
        }
    )

pd.set_option("display.max_colwidth", None)
df_final_comparison = pd.DataFrame(final_results).sort_values(by="Accuracy", ascending=False)

print("\n--- COMPARISON TABLE AFTER GA ---")
df_final_comparison.reset_index(drop=True, inplace=True)
print(df_final_comparison.to_string())
