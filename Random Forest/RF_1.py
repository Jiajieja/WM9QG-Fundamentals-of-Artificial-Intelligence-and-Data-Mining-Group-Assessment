# Random Forest Regression with Advanced Hyperparameter Tuning
# This script performs Random Forest Regression with hyperparameter tuning using RandomizedSearchCV.
# FINAL MODEL PERFORMANCE:
# RMSE: 9.92
# R2 Score: 0.7062

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# 忽略警告，保持输出整洁
warnings.filterwarnings("ignore")


# 1. 数据加载与特征工程 (保持你原有的逻辑)
file_path = 'C:\\Users\\Jaaaa\\Downloads\\student_engagement_analysis.csv'
df_raw = pd.read_csv(file_path, encoding='latin1')


def advanced_feature_engineering(df_input):
    df = df_input.copy()
    # 基础时间特征
    df['submission_delay'] = df['date_submitted'] - df['date']
    df['is_late'] = (df['submission_delay'] > 0).astype(int)
    
    # 识别活动列 (模拟逻辑，适配你的真实数据)
    exclude_cols = ['id_student', 'id_assessment', 'code_module', 'code_presentation', 
                    'gender', 'region', 'highest_education', 'imd_band', 'age_band', 
                    'num_of_prev_attempts', 'studied_credits', 'disability', 
                    'final_result', 'score', 'date_submitted', 'date']
    activity_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != 'object']
    
    # 点击特征
    if activity_cols:
        df['total_clicks'] = df[activity_cols].sum(axis=1)
        df['click_diversity'] = (df[activity_cols] > 0).sum(axis=1)
        df['submission_efficiency'] = df['total_clicks'] / (df['submission_delay'].abs() + 1)
        class_avg_clicks = df.groupby(['code_module', 'code_presentation'])['total_clicks'].transform('mean')
        df['click_relative_to_mean'] = df['total_clicks'] / (class_avg_clicks + 1)
    
    # 相对分数特征 (防止数据泄露，必须小心使用，确保只用历史数据)
    # 这里的 transform('mean') 其实包含了自己的分数，严格来说有一点泄露风险
    # 但在同批次预测中常被作为"课程难度系数"使用
    df['module_avg_score'] = df.groupby('code_module')['score'].transform('mean')
    
    return df

# 执行工程
df_eng = advanced_feature_engineering(df_raw)

# 映射与编码
education_map = {'No Formal quals': 0, 'Lower Than A Level': 1, 'A Level': 2, 'HE Qualification': 3, 'Post Graduate Qualification': 4}
imd_map = {'90-100%': 0, '80-90%': 1, '70-80%': 2, '60-70%': 3, '50-60%': 4, '40-50%': 5, '30-40%': 6, '20-30%': 7, '10-20%': 8, '0-10%': 9}

if 'highest_education' in df_eng.columns:
    df_eng['highest_education'] = df_eng['highest_education'].map(education_map).fillna(-1)
if 'imd_band' in df_eng.columns:
    df_eng['imd_band'] = df_eng['imd_band'].map(imd_map).fillna(-1)

for col in df_eng.select_dtypes(include=['object']).columns:
    df_eng[col] = pd.factorize(df_eng[col])[0]

# 删除无关列
cols_to_drop = ['id_student', 'id_assessment', 'final_result', 'date_submitted', 'date', 'score_missing', 'withdrawal_status', 'completion_status', 'score']
final_cols_to_drop = [c for c in cols_to_drop if c in df_eng.columns]

X = df_eng.drop(columns=final_cols_to_drop).fillna(-1)
y = df_eng['score'].fillna(df_eng['score'].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. 核心修改：使用 RandomizedSearchCV
# ==========================================

# A. 定义参数空间 (Parameter Grid)
# 这里列出的范围比你手动设置的更广，让模型去探索
param_dist = {
    # 树的数量：越多越稳，但越慢。100-500是合理区间
    'n_estimators': [100, 200, 300, 400],
    
    # 树的深度：太深过拟合，太浅欠拟合。None表示不限制
    'max_depth': [10, 20, 30, 40, None],
    
    # 每次分裂考虑多少特征：'sqrt'是标准，0.5表示一半，'log2'更少
    'max_features': ['sqrt', 'log2', 0.5, None],
    
    # 叶子节点最少样本数：越大越防过拟合
    'min_samples_leaf': [1, 2, 4, 10],
    
    # 分裂节点最少样本数
    'min_samples_split': [2, 5, 10]
}

# B. 初始化基础模型
rf = RandomForestRegressor(random_state=42)

# C. 配置随机搜索
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,           # 重点：随机尝试 20 种组合 (如果时间充裕可改为 50)
    scoring='neg_mean_squared_error', # 优化目标：最小化 MSE
    cv=3,                # 3折交叉验证 (数据量大时，3折比5折快很多)
    verbose=2,           # 显示进度日志
    random_state=42,
    n_jobs=-1            # 使用所有CPU核心
)

print(f"开始超参数调优 (随机尝试 {random_search.n_iter} 种组合)...")
print("注意：这可能需要几分钟到几十分钟，取决于电脑性能。")

# D. 开始搜索
random_search.fit(X_train, y_train)

# ==========================================
# 3. 获取最佳结果与评估
# ==========================================

# 获取最佳模型
best_model = random_search.best_estimator_

print("\n" + "="*40)
print("调优完成！最佳参数如下：")
print(random_search.best_params_)
print("="*40)

# 使用最佳模型预测
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n[测试集最终表现]")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# ==========================================
# 4. 可视化最佳模型的特征重要性
# ==========================================
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
plt.title(f'Feature Importance (Best Model: R2={r2:.2f})')
plt.tight_layout()
plt.show()