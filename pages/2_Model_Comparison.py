"""
pages/2_Model_Comparison.py  —  Linear Regression vs Random Forest
===================================================================
This page loads the pre-trained models (.pkl files saved by train_model.py)
and shows:
  - R², RMSE, MAE side by side
  - Actual vs Predicted scatter plots for both models
  - Residual distribution (are errors evenly spread?)
  - Feature importance (Random Forest)
  - A plain-language explanation of every metric

Beginner note on metrics:
  R²   = "How much of the variation does the model explain?" (1.0 = perfect)
  RMSE = "On average, how far off is each prediction (in dollars)?" (lower = better)
  MAE  = "On average, absolute dollar error per prediction"  (lower = better)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Model Comparison", page_icon="🤖", layout="wide")
st.title("🤖 Model Comparison: Linear Regression vs Random Forest")
st.markdown("Both models were trained on **80%** of the data. Metrics below are measured on the held-out **20% test set**.")

# ------------------------------------------------------------------
# Load data & models
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
    X = df_enc.drop("charges", axis=1)
    y = df_enc["charges"]
    return X, y

@st.cache_resource
def load_models():
    lr  = joblib.load("models/linear_regression.pkl")
    rf  = joblib.load("models/random_forest.pkl")
    sc  = joblib.load("models/scaler.pkl")
    return lr, rf, sc

X, y = load_data()
lr_model, rf_model, scaler = load_models()

# Reproduce the same train/test split used in training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_scaled = scaler.transform(X_test)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)

# Metrics helper
def get_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

lr_r2, lr_rmse, lr_mae = get_metrics(y_test, y_pred_lr)
rf_r2, rf_rmse, rf_mae = get_metrics(y_test, y_pred_rf)

# ------------------------------------------------------------------
# 1. METRIC CARDS
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("1. Performance Metrics (Test Set)")

col_lr, col_rf = st.columns(2)

with col_lr:
    st.markdown("### Linear Regression")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{lr_r2:.4f}")
    c2.metric("RMSE",     f"${lr_rmse:,.0f}")
    c3.metric("MAE",      f"${lr_mae:,.0f}")

with col_rf:
    st.markdown("### Random Forest  🏆")
    c1, c2, c3 = st.columns(3)
    delta_r2   = rf_r2 - lr_r2
    delta_rmse = lr_rmse - rf_rmse   # positive = RF is lower (better)
    delta_mae  = lr_mae - rf_mae
    c1.metric("R² Score", f"{rf_r2:.4f}", delta=f"+{delta_r2:.4f} vs LR")
    c2.metric("RMSE",     f"${rf_rmse:,.0f}", delta=f"-${delta_rmse:,.0f} vs LR", delta_color="inverse")
    c3.metric("MAE",      f"${rf_mae:,.0f}",  delta=f"-${delta_mae:,.0f} vs LR",  delta_color="inverse")

# Plain-language explanations
with st.expander("What do these metrics mean? (Beginner explanation)"):
    st.markdown("""
| Metric | Meaning | Good value |
|--------|---------|------------|
| **R² Score** | "How much of the variation in charges does the model explain?" 0 = useless, 1 = perfect. R²=0.87 means 87% explained. | Closer to 1.0 |
| **RMSE** (Root Mean Squared Error) | Average prediction error in dollars. Penalises big mistakes more than small ones. | As low as possible |
| **MAE** (Mean Absolute Error) | Average absolute dollar difference between predicted and actual. Easier to interpret than RMSE. | As low as possible |
""")

# ------------------------------------------------------------------
# 2. COMPARISON BAR CHART
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("2. Side-by-Side Metric Comparison")

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        x=["Linear Regression", "Random Forest"],
        y=[lr_r2, rf_r2],
        color=["Linear Regression", "Random Forest"],
        color_discrete_sequence=["#f59e0b", "#16a34a"],
        title="R² Score (higher = better)",
        labels={"x": "Model", "y": "R² Score"},
        text=[f"{lr_r2:.4f}", f"{rf_r2:.4f}"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="RMSE", x=["Linear Regression", "Random Forest"],
                          y=[lr_rmse, rf_rmse], marker_color=["#f59e0b", "#16a34a"],
                          text=[f"${lr_rmse:,.0f}", f"${rf_rmse:,.0f}"],
                          textposition="outside"))
    fig.update_layout(title="RMSE in $ (lower = better)",
                      yaxis_title="RMSE ($)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 3. ACTUAL vs PREDICTED SCATTER
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("3. Actual vs Predicted Charges")
st.markdown(
    "Each dot is a test record. The **diagonal line** = perfect prediction. "
    "Dots close to the line = accurate. Dots far away = large errors."
)

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        x=y_test, y=y_pred_lr,
        labels={"x": "Actual Charges ($)", "y": "Predicted Charges ($)"},
        title="Linear Regression — Actual vs Predicted",
        opacity=0.5, color_discrete_sequence=["#f59e0b"],
    )
    # Perfect prediction line
    max_val = max(y_test.max(), y_pred_lr.max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                             mode="lines", name="Perfect", line=dict(color="black", dash="dash")))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        x=y_test, y=y_pred_rf,
        labels={"x": "Actual Charges ($)", "y": "Predicted Charges ($)"},
        title="Random Forest — Actual vs Predicted",
        opacity=0.5, color_discrete_sequence=["#16a34a"],
    )
    max_val = max(y_test.max(), float(y_pred_rf.max()))
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                             mode="lines", name="Perfect", line=dict(color="black", dash="dash")))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 4. RESIDUAL PLOTS  (errors should be randomly scattered around 0)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("4. Residuals (Prediction Errors)")
st.markdown(
    "**Residual = Actual − Predicted.** "
    "Ideally residuals should be randomly scattered around zero — no pattern."
)

col1, col2 = st.columns(2)
res_lr = y_test.values - y_pred_lr
res_rf = y_test.values - y_pred_rf

with col1:
    fig = px.histogram(res_lr, nbins=30, title="Linear Regression — Residual Distribution",
                       color_discrete_sequence=["#f59e0b"],
                       labels={"value": "Residual ($)", "count": "Count"})
    fig.add_vline(x=0, line_color="black", line_dash="dash")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(res_rf, nbins=30, title="Random Forest — Residual Distribution",
                       color_discrete_sequence=["#16a34a"],
                       labels={"value": "Residual ($)", "count": "Count"})
    fig.add_vline(x=0, line_color="black", line_dash="dash")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Both histograms are roughly bell-shaped and centred near zero — "
    "a sign that neither model is systematically over- or under-predicting."
)

# ------------------------------------------------------------------
# 5. FEATURE IMPORTANCE  (Random Forest)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("5. Feature Importance (Random Forest)")
st.markdown(
    "Random Forest tells us **which input features it relied on most** when making predictions. "
    "Higher importance = more influence on the predicted charge."
)

importances = rf_model.feature_importances_
feat_names  = X.columns.tolist()
feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)

fig = px.bar(
    feat_df, x="Importance", y="Feature", orientation="h",
    title="Random Forest Feature Importances",
    color="Importance", color_continuous_scale="Greens",
    text=feat_df["Importance"].apply(lambda x: f"{x:.3f}"),
)
fig.update_layout(coloraxis_showscale=False, yaxis_title="")
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "**smoker_yes** is overwhelmingly the most important feature — "
    "it alone accounts for the largest share of the variation in charges. "
    "Age and BMI follow."
)

# ------------------------------------------------------------------
# 6. VERDICT
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("6. Verdict")
st.success(
    f"**Random Forest** is the recommended model.\n\n"
    f"- R² = **{rf_r2:.4f}** vs {lr_r2:.4f} for LR\n"
    f"- RMSE = **${rf_rmse:,.0f}** vs ${lr_rmse:,.0f} for LR\n\n"
    "Random Forest handles the non-linear relationship between smoking+BMI and charges "
    "better than a single linear equation can."
)
