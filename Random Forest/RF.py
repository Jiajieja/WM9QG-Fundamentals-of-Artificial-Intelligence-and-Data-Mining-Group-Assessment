import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

studInfo=pd.read_csv("Original Dataset\\studentInfo.csv")
assessments=pd.read_csv("Original Dataset\\assessments.csv")
studAss=pd.read_csv("Original Dataset\\studentAssessment.csv")
studVle=pd.read_csv("Original Dataset\\studentVle.csv")
vle=pd.read_csv("Original Dataset\\vle.csv")
exams=assessments[assessments["assessment_type"]=="Exam"]
others=assessments[assessments["assessment_type"]!="Exam"]
amounts=others.groupby(["code_module","code_presentation"]).count()["id_assessment"] 
amounts=amounts.reset_index()
# print(amounts.head())

def pass_fail(grade):
    if grade>=40:
        return True
    else:
        return False
stud_ass=pd.merge(studAss,others,how="inner",on=["id_assessment"])
stud_ass["pass"]=stud_ass["score"].apply(pass_fail)
stud_ass["weighted_grade"]=stud_ass["score"]*stud_ass["weight"]/100
# print(stud_ass.head())

avg_grade=stud_ass.groupby(["id_student","code_module","code_presentation"]).sum()["weighted_grade"].reset_index()
# print(avg_grade.head())

pass_rate=pd.merge((stud_ass[stud_ass["pass"]==True].groupby(["id_student","code_module","code_presentation"]).count()["pass"]).reset_index(),amounts,how="left",on=["code_module","code_presentation"])
pass_rate["pass_rate"]=pass_rate["pass"]/pass_rate["id_assessment"]
pass_rate.drop(["pass","id_assessment"], axis=1,inplace=True)
# print(pass_rate.head())

stud_exams=pd.merge(studAss,exams,how="inner",on=["id_assessment"])
stud_exams["exam_score"]=stud_exams["score"]
stud_exams.drop(["id_assessment","date_submitted","is_banked", "score","assessment_type","date","weight"],axis=1,inplace=True)
# print(stud_exams.head())

# print(vle.head())

vle[~vle["week_from"].isna()]

# print(studVle.head())

avg_per_site=studVle.groupby(["id_student","id_site","code_module","code_presentation"]).mean().reset_index()
# print(avg_per_site.head())

avg_per_student=avg_per_site.groupby(["id_student","code_module","code_presentation"]).mean()[["date","sum_click"]].reset_index()
# print(avg_per_student.head())

studInfo=studInfo[studInfo["final_result"]!="Withdrawn"]
studInfo=studInfo[["code_module","code_presentation","id_student","num_of_prev_attempts","final_result"]]
# print(studInfo.head())

df_1=pd.merge(avg_grade,pass_rate,how="inner",on=["id_student","code_module","code_presentation"])
assessment_info=pd.merge(df_1, stud_exams, how="inner", on=["id_student","code_module","code_presentation"])
# print(assessment_info.head())

df_2=pd.merge(studInfo,assessment_info,how="inner",on=["id_student","code_module","code_presentation"])
final_df=pd.merge(df_2,avg_per_student,how="inner", on=["id_student","code_module","code_presentation"])
final_df.drop(["id_student","code_module","code_presentation"],axis=1,inplace=True)
# print(final_df.head())

# print(final_df.describe())
# print(final_df.info())

# plt.figure(figsize=(8,6))
# sns.heatmap(final_df.corr(),annot=True)
# plt.savefig("correlation_matrix.png")

plt.figure(figsize=(8,6))
sns.countplot(data=final_df, x="final_result")
plt.savefig("Random Forest/final_result_distribution.png")

g = sns.pairplot(final_df)
g.savefig("Random Forest/pairplot.png")

# print(final_df[final_df["sum_click"]>10])
# print(final_df[final_df["num_of_prev_attempts"]>4])
final_df=final_df[final_df["sum_click"]<=10]
final_df=final_df[final_df["num_of_prev_attempts"]<=4]
print(final_df.head())

# Splitting the data
# ---------------------------------------------------
X=final_df.drop("final_result", axis=1)
y=final_df["final_result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Creating different feature sets
# ---------------------------------------------------
X1_test=X_test
X1_train=X_train
# Remove weighted_grade for testing
# ---------------------------------------------------
X2_test=X_test.drop("weighted_grade",axis=1)
X2_train=X_train.drop("weighted_grade",axis=1)
# Remove pass_rate for testing
# ---------------------------------------------------
X3_test=X_test.drop("pass_rate",axis=1)
X3_train=X_train.drop("pass_rate",axis=1)

# Scaling the features
scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
scaler3=MinMaxScaler()

# Fit and transform the training data, transform the testing data
X1_train=scaler1.fit_transform(X1_train)
X1_test=scaler1.transform(X1_test)
X2_train=scaler2.fit_transform(X2_train)
X2_test=scaler2.transform(X2_test)
X3_train=scaler3.fit_transform(X3_train)
X3_test=scaler3.transform(X3_test)

# training Random Forest Classifier
# ---------------------------------------------------
rf1=RandomForestClassifier(n_estimators=300)
rf1.fit(X1_train,y_train)
result_rf1=rf1.predict(X1_test)
print("\n")
print(classification_report(y_test,result_rf1))
# ---------------------------------------------------
rf2=RandomForestClassifier(n_estimators=300)
rf2.fit(X2_train,y_train)
result_rf2=rf2.predict(X2_test)
print("\n")
print(classification_report(y_test,result_rf2))
# ---------------------------------------------------
rf3=RandomForestClassifier(n_estimators=300)
rf3.fit(X3_train,y_train)
result_rf3=rf3.predict(X3_test)
print("\n")
print(classification_report(y_test,result_rf3))

# Calculating Accuracies
acc1 = accuracy_score(y_test, result_rf1)
acc2 = accuracy_score(y_test, result_rf2)
acc3 = accuracy_score(y_test, result_rf3)

print("Model Accuracies:")
print(f"RF1 (all features): {acc1:.4f}")
print(f"RF2 (no weighted_grade): {acc2:.4f}")
print(f"RF3 (no pass_rate): {acc3:.4f}")

# Selecting the Best Model
accuracies = [acc1, acc2, acc3]
models = [rf1, rf2, rf3]
feature_sets = [X_train.columns, 
                X_train.drop("weighted_grade", axis=1).columns,
                X_train.drop("pass_rate", axis=1).columns]

best_index = np.argmax(accuracies)
best_model = models[best_index]
best_features = feature_sets[best_index]

print("\nBest Model is: RF" + str(best_index+1))
print("Accuracy:", accuracies[best_index])

# Plotting Feature Importance of the Best Model
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), best_features[indices], rotation=90)
plt.title("Feature Importance of Best Random Forest Model")
plt.tight_layout()
plt.savefig("Random Forest/best_model_feature_importance.png")
