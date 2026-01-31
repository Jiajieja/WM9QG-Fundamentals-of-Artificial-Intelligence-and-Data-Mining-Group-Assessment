import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
studInfo = pd.read_csv("studentInfo.csv")
assessments = pd.read_csv("assessments.csv")
studAss = pd.read_csv("studentAssessment.csv")
studVle = pd.read_csv("studentVle.csv")
vle = pd.read_csv("vle.csv")

# ---------------------------------------------------
# Handle missing values
# ---------------------------------------------------
studInfo["imd_band"] = studInfo["imd_band"].fillna(studInfo["imd_band"].mode()[0])
studAss["score"] = studAss["score"].fillna(0)
assessments["date"] = assessments["date"].fillna(assessments["date"].median())
vle["week_from"] = vle["week_from"].fillna(-1)
vle["week_to"] = vle["week_to"].fillna(-1)

# ---------------------------------------------------
# Feature engineering
# ---------------------------------------------------
exams = assessments[assessments["assessment_type"] == "Exam"]
others = assessments[assessments["assessment_type"] != "Exam"]

amounts = others.groupby(
    ["code_module", "code_presentation"]
).count()["id_assessment"].reset_index()

def pass_fail(score):
    return score >= 40

stud_ass = pd.merge(studAss, others, on="id_assessment")
stud_ass["pass"] = stud_ass["score"].apply(pass_fail)
stud_ass["weighted_grade"] = stud_ass["score"] * stud_ass["weight"] / 100

avg_grade = stud_ass.groupby(
    ["id_student", "code_module", "code_presentation"]
).sum()["weighted_grade"].reset_index()

pass_rate = pd.merge(
    stud_ass[stud_ass["pass"] == True]
    .groupby(["id_student", "code_module", "code_presentation"])
    .count()["pass"]
    .reset_index(),
    amounts,
    on=["code_module", "code_presentation"]
)

pass_rate["pass_rate"] = pass_rate["pass"] / pass_rate["id_assessment"]
pass_rate.drop(["pass", "id_assessment"], axis=1, inplace=True)

stud_exams = pd.merge(studAss, exams, on="id_assessment")
stud_exams["exam_score"] = stud_exams["score"]
stud_exams.drop(
    ["id_assessment", "date_submitted", "is_banked", "score",
     "assessment_type", "date", "weight"],
    axis=1,
    inplace=True
)

avg_per_site = studVle.groupby(
    ["id_student", "id_site", "code_module", "code_presentation"]
).mean().reset_index()

avg_per_student = avg_per_site.groupby(
    ["id_student", "code_module", "code_presentation"]
).mean()[["date", "sum_click"]].reset_index()

studInfo = studInfo[studInfo["final_result"] != "Withdrawn"]
studInfo = studInfo[
    ["code_module", "code_presentation", "id_student",
     "num_of_prev_attempts", "final_result"]
]

df_1 = pd.merge(avg_grade, pass_rate,
                on=["id_student", "code_module", "code_presentation"])

assessment_info = pd.merge(
    df_1, stud_exams,
    on=["id_student", "code_module", "code_presentation"]
)

df_2 = pd.merge(
    studInfo, assessment_info,
    on=["id_student", "code_module", "code_presentation"]
)

final_df = pd.merge(
    df_2, avg_per_student,
    on=["id_student", "code_module", "code_presentation"]
)

final_df.drop(
    ["id_student", "code_module", "code_presentation"],
    axis=1,
    inplace=True
)

# ---------------------------------------------------
# Simple filtering
# ---------------------------------------------------
final_df = final_df[final_df["sum_click"] <= 10]
final_df = final_df[final_df["num_of_prev_attempts"] <= 4]

# ---------------------------------------------------
# Train / Test split
# ---------------------------------------------------
X = final_df.drop("final_result", axis=1)
y = final_df["final_result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Scaling (X1 only)
# ---------------------------------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# Baseline model (Before tuning)
# ---------------------------------------------------
baseline_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

baseline_model.fit(X_train_scaled, y_train)
y_pred_baseline = baseline_model.predict(X_test_scaled)

print("Baseline model (before tuning)")
print(classification_report(y_test, y_pred_baseline))

# ---------------------------------------------------
# Confusion Matrix — Baseline model
# ---------------------------------------------------
cm_baseline = confusion_matrix(y_test, y_pred_baseline)

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(
    cm_baseline,
    display_labels=baseline_model.classes_
).plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix (Before Tuning)")
plt.savefig("Random Forest/confusion_matrix_baseline.png")

# ---------------------------------------------------
# Optuna + Stratified CV
# ---------------------------------------------------
def objective(trial):

    try:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }

        model = RandomForestClassifier(**params)

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring="average_precision",
            error_score=0.0
        )

        return scores.mean()

    except Exception:
        return 0.0

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)

# ---------------------------------------------------
# Train tuned model
# ---------------------------------------------------
best_model = RandomForestClassifier(
    **study.best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=best_model.classes_
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Tuned Random Forest)")
plt.savefig("Random Forest/confusion_matrix_tuned_labeled.png")

# ---------------------------------------------------
# Binary reformulation: Fail vs Non-Fail
# ---------------------------------------------------
y_test_binary = y_test.apply(lambda x: 1 if x == "Fail" else 0)
fail_index = list(best_model.classes_).index("Fail")
y_prob_fail = best_model.predict_proba(X_test_scaled)[:, fail_index]

# ---------------------------------------------------
# ROC–AUC
# ---------------------------------------------------
fpr, tpr, _ = roc_curve(y_test_binary, y_prob_fail)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC (Fail vs Non-Fail)")
plt.legend()
plt.savefig("Random Forest/roc_auc_tuned.png")

# ---------------------------------------------------
# Precision–Recall Curve
# ---------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test_binary, y_prob_fail)
ap = average_precision_score(y_test_binary, y_prob_fail)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Fail vs Non-Fail)")
plt.legend()
plt.savefig("Random Forest/precision_recall_curve_tuned.png")
