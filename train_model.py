"""
train_model.py
==============
Trains two models on insurance.csv:
  1. Linear Regression
  2. Random Forest Regressor

Then saves them as .pkl files inside the models/ folder so the
Streamlit app can load them instantly without retraining every time.

Beginner explanation:
  - We use scikit-learn, which is a Python library full of ready-made
    machine learning algorithms.
  - "Fitting" a model means feeding it the training data so it can
    learn the patterns.
  - "Pickling" (joblib.dump) means saving the trained model to a file
    on disk — like saving a Word document — so we can re-open it later.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
# pd.read_csv reads the spreadsheet-like file into a DataFrame (table).
df = pd.read_csv("insurance.csv")
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ------------------------------------------------------------------
# 2. ENCODE CATEGORICAL COLUMNS
# ------------------------------------------------------------------
# Machine learning models only understand numbers, not text like
# "male" or "southwest". So we convert text to 0s and 1s.
#
# pd.get_dummies creates a new binary column for each category.
# drop_first=True removes one column to avoid redundancy
# (if sex_male=0, we already know it's female — no need for sex_female).
df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

# Rename columns for clarity
df_encoded.rename(columns={
    "sex_male":        "sex_male",
    "smoker_yes":      "smoker_yes",
    "region_northwest":"region_northwest",
    "region_southeast":"region_southeast",
    "region_southwest":"region_southwest",
}, inplace=True)

print("Encoded columns:", df_encoded.columns.tolist())

# ------------------------------------------------------------------
# 3. SPLIT FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------------
# X = all columns except "charges" — these are the inputs (features).
# y = the "charges" column — this is what we want to predict (target).
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# ------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ------------------------------------------------------------------
# We hold out 20% of the data (test set) so we can evaluate how
# well the model performs on data it has never seen before.
# random_state=42 makes the split reproducible (same split every run).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ------------------------------------------------------------------
# 5. SCALE FEATURES (for Linear Regression)
# ------------------------------------------------------------------
# StandardScaler normalises features so they all have a mean of 0
# and standard deviation of 1. This helps Linear Regression converge
# faster and prevents large-valued features (like charges ~$13k)
# from dominating small ones (like children 0-5).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std from train, then scale
X_test_scaled  = scaler.transform(X_test)         # use SAME mean/std to scale test

# ------------------------------------------------------------------
# 6. TRAIN LINEAR REGRESSION
# ------------------------------------------------------------------
# Linear Regression fits a straight-line formula:
#   charges = a*age + b*bmi + c*children + ... + intercept
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
lr_r2   = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae  = mean_absolute_error(y_test, y_pred_lr)
print(f"\nLinear Regression  ->  R2={lr_r2:.4f}  RMSE=${lr_rmse:,.2f}  MAE=${lr_mae:,.2f}")

# ------------------------------------------------------------------
# 7. TRAIN RANDOM FOREST
# ------------------------------------------------------------------
# Random Forest builds many decision trees (n_estimators=100) and
# averages their predictions. It handles non-linear patterns well.
# max_depth=10 limits how deep each tree grows (prevents overfitting).
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)   # RF doesn't need scaled data

y_pred_rf = rf_model.predict(X_test)
rf_r2   = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae  = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest      ->  R2={rf_r2:.4f}  RMSE=${rf_rmse:,.2f}  MAE=${rf_mae:,.2f}")

# ------------------------------------------------------------------
# 8. SAVE MODELS TO DISK
# ------------------------------------------------------------------
# joblib.dump saves the trained model as a binary file (.pkl).
# The Streamlit app will load these files with joblib.load()
# so users get instant predictions without waiting for training.
os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, "models/linear_regression.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(scaler,   "models/scaler.pkl")

# Also save the feature column names (needed to align user input)
feature_cols = X.columns.tolist()
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("\nModels saved to models/ folder:")
print("   models/linear_regression.pkl")
print("   models/random_forest.pkl")
print("   models/scaler.pkl")
print("   models/feature_cols.pkl")
