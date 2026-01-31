# Open University Learning Analytics Dataset (OULAD) – Student Performance Classification

## 📓 Main Analysis File

- **OULA_DATASET_ANALYSIS (CLASSIFICATION).ipynb**  
Contains the complete end-to-end workflow, including data cleaning, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and performance evaluation for classification.

## 📌 Project Overview

The **Open University Learning Analytics Dataset (OULAD)** contains anonymized data on students enrolled in online courses offered by the Open University. The dataset is composed of multiple interconnected tables capturing student demographics, academic performance, course structure, and online learning behavior.

It includes information such as:

* Student registration and withdrawal dates
* Assessment details and scores
* Final academic results
* Prior educational background
* Interaction logs from the Virtual Learning Environment (VLE)

Together, these components provide a comprehensive view of student engagement and progression throughout online modules, enabling detailed analysis of learning behavior, performance patterns, and factors influencing student success and withdrawal.

---

## 🧹 Data Preparation & Missing Value Handling

This phase focused on cleaning and preparing the raw data to ensure reliability and consistency for downstream analysis.

### Initial Inspection

* Examined the dimensions and structure of all seven datasets:

  * `studentregistration`
  * `studentinfo`
  * `studentvle`
  * `studentassessment`
  * `courses`
  * `vle`
  * `assessments`

### Missing Value Identification

* Performed a comprehensive scan to identify missing values across all columns in each dataset.

### Missing Value Treatment

* **studentinfo**

  * Missing `imd_band` values were imputed using the mode to preserve distributional characteristics.
* **studentregistration**

  * Missing `date_registration` values were filled using the median.
  * Missing `date_unregistration` values were replaced with `-1` to explicitly represent students who did not withdraw.
* **studentassessment**

  * Missing `score` values were set to `0`, assuming unrecorded assessments correspond to zero performance.
* **assessments**

  * Missing `date` values were imputed using the median for temporal consistency.
* **vle**

  * Missing `week_from` and `week_to` values were replaced with `-1` to denote undefined periods.

---

## 📊 Exploratory Data Analysis (EDA)

EDA was conducted to understand distributions, relationships, and patterns within the prepared dataset.

### Categorical Variable Analysis

* **Count plots** were used to examine distributions of:

  * `final_result`
  * `highest_education`
  * `age_band`
  * `gender`
  * `disability`
  * `imd_band`
  * `num_of_prev_attempts`
  * `code_module`
* **Pie charts** visualized proportions of:

  * Gender
  * Disability
  * Code presentation
  * Age band
* A dedicated **count plot** explored student distribution across regions.

### Numerical Variable Analysis

* Histograms with KDE were generated for:

  * `total_clicks`
  * `studied_credits`
  * `avg_score`
  * `date_registration`

### Relationship Analysis

* **Final Result vs. Features**

  * Box plots for `total_clicks` and `avg_score` across `final_result`
  * Count plots for `highest_education` by `final_result`
* **Cross-Tabulations**

  * Normalized crosstabs between:

    * `highest_education` and `final_result`
    * `imd_band` and `final_result`
* **Correlation Analysis**

  * Heatmap of numerical features:

    * `num_of_prev_attempts`
    * `studied_credits`
    * `total_score`
    * `avg_score`
    * `num_assessments`
    * `total_clicks`

### Specialized Visualizations

* **Engagement Arc**

  * Line plot of average daily clicks over time, segmented by final result.
* **Effort–Performance Map**

  * Scatter plot of `total_clicks` vs. `avg_score`, colored by final result.
* **Socio-Economic Ladder**

  * Stacked bar chart showing final results across `imd_band`.

### Course & Registration Insights

* Average assessment scores by region.
* Distribution of registration timing (`date_registration`) relative to course start.

---

## 🧠 Feature Engineering

Raw variables were transformed into meaningful features to enhance predictive performance.

### Time-Based Features

* `days_before_start`: Days between registration and course start.
* `has_withdrawn`: Binary flag indicating course withdrawal.
* `days_until_withdrawal`: Duration between registration and withdrawal (or `-1` if not withdrawn).

### Numerical Encoding of Ordinal Variables

* `imd_num`: Numerical midpoint of `imd_band`.
* `age_num`: Midpoint mapping of `age_band`.
* `edu_level`: Ordinal encoding of `highest_education` (0–4).

### Binary Encoding

* `gender_m`: Male = 1, Female = 0.
* `disability_flag`: Yes = 1, No = 0.

### Interaction Features

* `clicks_per_credit`: Engagement intensity relative to course load.
* `score_per_assess`: Average score per assessment.

### Target Variable

* `target_pass`: Binary classification target

  * 1 → Pass or Distinction
  * 0 → Fail or Withdrawn

All infinite or missing numerical values generated during feature construction were replaced with `0`.

---

## 🔢 Encoding Categorical Variables

Remaining categorical features were converted into numeric form using **Label Encoding**:

* `code_module`
* `code_presentation`
* `region`

A final `modeling_df` was constructed containing:

* Selected raw features
* Engineered features
* Encoded categorical variables
* Target variable (`target_pass`)

---

## 🔀 Train–Test Split

* Features (`X`) excluded `id_student` and `target_pass`.
* Target (`y_class`) set as `target_pass`.
* Data split using `train_test_split`:

  * 80% training, 20% testing
  * `random_state = 42` for reproducibility
  * `stratify = y_class` to preserve class distribution

---

## 🤖 Model Selection & Performance Comparison

Four classification models were trained and evaluated:

* **Random Forest Classifier**
* **Decision Tree Classifier**
* **XGBoost Classifier**
* **Support Vector Machine (LinearSVC)** with feature scaling

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Specificity
* Confusion Matrix (visualized via heatmaps)

### Feature Importance

* Tree-based models: native feature importance extraction
* Linear SVM: absolute coefficient values

---

## ⚙️ Hyperparameter Tuning & Advanced Evaluation

### Hyperparameter Optimization

* GridSearchCV applied to all models
* 3-fold cross-validation (`cv = 3`)
* Accuracy used as the optimization metric
* Best hyperparameters and cross-validation scores identified

### Optimized Model Evaluation

* Optimized models evaluated on test data
* Metrics compared against baseline versions

### Performance Curves

* **ROC–AUC Curves** to assess discrimination ability
* **Precision–Recall Curves** to evaluate positive-class performance on imbalanced data

### Consolidated Performance Summary

* A comprehensive performance table comparing baseline vs. optimized models
* Multi-facet visual comparison across Accuracy, Precision, Recall, and F1-Score

---

## 🏆 Final Conclusion

Based on a comprehensive evaluation—particularly emphasizing **F1-Score** to balance precision and recall—the **optimized XGBoost classifier** emerged as the best-performing model for predicting student outcomes in the OULAD dataset.

---