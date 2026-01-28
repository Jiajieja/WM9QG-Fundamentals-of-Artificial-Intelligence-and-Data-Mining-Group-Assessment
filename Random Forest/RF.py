import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
studInfo = pd.read_csv("studentInfo.csv")
assessments = pd.read_csv("assessments.csv")
studAss = pd.read_csv("studentAssessment.csv")
studVle = pd.read_csv("studentVle.csv")
vle = pd.read_csv("vle.csv")

exams = assessments[assessments["assessment_type"] == "Exam"]
others = assessments[assessments["assessment_type"] != "Exam"]

amounts = others.groupby(["code_module","code_presentation"]).count()["id_assessment"].reset_index()

def pass_fail(grade):
    return grade >= 40

stud_ass = pd.merge(studAss, others, how="inner", on="id_assessment")
stud_ass["pass"] = stud_ass["score"].apply(pass_fail)
stud_ass["weighted_grade"] = stud_ass["score"] * stud_ass["weight"] / 100

avg_grade = stud_ass.groupby(
    ["id_student","code_module","code_presentation"]
).sum()["weighted_grade"].reset_index()

pass_rate = pd.merge(
    stud_ass[stud_ass["pass"] == True]
    .groupby(["id_student","code_module","code_presentation"])
    .count()["pass"]
    .reset_index(),
    amounts,
    how="left",
    on=["code_module","code_presentation"]
)

pass_rate["pass_rate"] = pass_rate["pass"] / pass_rate["id_assessment"]
pass_rate.drop(["pass","id_assessment"], axis=1, inplace=True)

stud_exams = pd.merge(studAss, exams, how="inner", on="id_assessment")
stud_exams["exam_score"] = stud_exams["score"]
stud_exams.drop(
    ["id_assessment","date_submitted","is_banked","score",
     "assessment_type","date","weight"],
    axis=1,
    inplace=True
)

avg_per_site = studVle.groupby(
    ["id_student","id_site","code_module","code_presentation"]
).mean().reset_index()

avg_per_student = avg_per_site.groupby(
    ["id_student","code_module","code_presentation"]
).mean()[["date","sum_click"]].reset_index()

studInfo = studInfo[studInfo["final_result"] != "Withdrawn"]
studInfo = studInfo[[
    "code_module","code_presentation","id_student",
    "num_of_prev_attempts","final_result"
]]

df_1 = pd.merge(avg_grade, pass_rate,
                on=["id_student","code_module","code_presentation"])

assessment_info = pd.merge(
    df_1, stud_exams,
    on=["id_student","code_module","code_presentation"]
)

df_2 = pd.merge(
    studInfo, assessment_info,
    on=["id_student","code_module","code_presentation"]
)

final_df = pd.merge(
    df_2, avg_per_student,
    on=["id_student","code_module","code_presentation"]
)

final_df.drop(
    ["id_student","code_module","code_presentation"],
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
    X, y, test_size=0.3, random_state=42
)

# Feature sets
X1_train, X1_test = X_train, X_test
X2_train, X2_test = X_train.drop("weighted_grade", axis=1), X_test.drop("weighted_grade", axis=1)
X3_train, X3_test = X_train.drop("pass_rate", axis=1), X_test.drop("pass_rate", axis=1)

# Scaling
scaler1, scaler2, scaler3 = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
X1_train = scaler1.fit_transform(X1_train)
X1_test  = scaler1.transform(X1_test)
X2_train = scaler2.fit_transform(X2_train)
X2_test  = scaler2.transform(X2_test)
X3_train = scaler3.fit_transform(X3_train)
X3_test  = scaler3.transform(X3_test)

# ---------------------------------------------------
# Train models
# ---------------------------------------------------
rf1 = RandomForestClassifier(n_estimators=300, random_state=42)
rf2 = RandomForestClassifier(n_estimators=300, random_state=42)
rf3 = RandomForestClassifier(n_estimators=300, random_state=42)

rf1.fit(X1_train, y_train)
rf2.fit(X2_train, y_train)
rf3.fit(X3_train, y_train)

pred1 = rf1.predict(X1_test)
pred2 = rf2.predict(X2_test)
pred3 = rf3.predict(X3_test)

accs = [
    accuracy_score(y_test, pred1),
    accuracy_score(y_test, pred2),
    accuracy_score(y_test, pred3)
]

best_index = np.argmax(accs)
models = [rf1, rf2, rf3]
preds  = [pred1, pred2, pred3]
X_tests = [X1_test, X2_test, X3_test]

best_model = models[best_index]
y_pred_best = preds[best_index]
X_test_best = X_tests[best_index]

print("Best Model: RF", best_index + 1)
print(classification_report(y_test, y_pred_best))

# ---------------------------------------------------
# Confusion Matrix (multiclass)
# ---------------------------------------------------
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6,5))
ConfusionMatrixDisplay(cm).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Multiclass)")
plt.savefig("Random Forest/confusion_matrix.png")

# ---------------------------------------------------
# Binary reformulation: Fail vs Non-Fail
# ---------------------------------------------------
y_test_binary = y_test.apply(lambda x: 1 if x == "Fail" else 0)

fail_index = list(best_model.classes_).index("Fail")
y_prob_fail = best_model.predict_proba(X_test_best)[:, fail_index]

# ---------------------------------------------------
# ROC–AUC
# ---------------------------------------------------
fpr, tpr, _ = roc_curve(y_test_binary, y_prob_fail)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC (Fail vs Non-Fail)")
plt.legend()
plt.savefig("Random Forest/roc_auc_curve.png")

# ---------------------------------------------------
# Precision–Recall Curve
# ---------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test_binary, y_prob_fail)
ap_score = average_precision_score(y_test_binary, y_prob_fail)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"AP = {ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Fail vs Non-Fail)")
plt.legend()
plt.savefig("Random Forest/precision_recall_curve.png")
