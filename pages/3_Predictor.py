"""
pages/3_Predictor.py  —  Interactive Prediction Tool
=====================================================
This is the "product" page: users enter their personal details
and get an instant estimated insurance cost from the trained model.

How it works step by step:
  1. User fills in sliders / dropdowns in the sidebar
  2. We build a one-row DataFrame matching the training feature format
  3. We pass it to the loaded model's .predict() method
  4. We display the result with a breakdown of cost drivers

Beginner note on st.sidebar:
  Any st.XXX call inside `with st.sidebar:` appears in the left panel
  instead of the main page. Great for input controls!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Predict Your Insurance Charges")
st.markdown("Fill in your details in the **sidebar** and click **Predict** to get your estimate.")

# ------------------------------------------------------------------
# Load models  (st.cache_resource keeps them in memory between runs)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    lr  = joblib.load("models/linear_regression.pkl")
    rf  = joblib.load("models/random_forest.pkl")
    sc  = joblib.load("models/scaler.pkl")
    fc  = joblib.load("models/feature_cols.pkl")
    return lr, rf, sc, fc

lr_model, rf_model, scaler, feature_cols = load_models()

# ------------------------------------------------------------------
# SIDEBAR — User inputs
# ------------------------------------------------------------------
st.sidebar.header("Enter Your Details")

age = st.sidebar.slider("Age", min_value=18, max_value=64, value=30, step=1,
                         help="Your current age in years")

sex = st.sidebar.radio("Sex", options=["Female", "Male"],
                        help="Biological sex as used in insurance pricing")

bmi = st.sidebar.slider("BMI (Body Mass Index)", min_value=10.0, max_value=55.0,
                          value=25.0, step=0.1, format="%.1f",
                          help="BMI = weight(kg) / height(m)²")

children = st.sidebar.selectbox("Number of Children / Dependents",
                                  options=[0, 1, 2, 3, 4, 5], index=0,
                                  help="Number of dependents covered by the insurance")

smoker = st.sidebar.radio("Smoker?", options=["No", "Yes"],
                           help="Whether you smoke tobacco regularly")

region = st.sidebar.selectbox("US Region", options=["Southwest", "Southeast", "Northwest", "Northeast"],
                                help="The US region where you reside")

model_choice = st.sidebar.radio(
    "Choose Model",
    options=["Random Forest (Recommended)", "Linear Regression"],
    help="Random Forest is more accurate (R²=0.87 vs 0.78)"
)

predict_btn = st.sidebar.button("🔮 Predict Charges", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Build input feature row
# ------------------------------------------------------------------
def build_input(age, sex, bmi, children, smoker, region, feature_cols):
    """
    Converts the user's inputs into the same format the model was trained on.
    We create a DataFrame with all required columns, setting dummy columns
    (like sex_male, smoker_yes, region_northwest) to 0 or 1.
    """
    data = {col: 0 for col in feature_cols}   # start with all zeros
    data["age"]      = age
    data["bmi"]      = bmi
    data["children"] = children
    if sex == "Male":
        data["sex_male"] = 1
    if smoker == "Yes":
        data["smoker_yes"] = 1
    if region == "Northwest":
        data["region_northwest"] = 1
    elif region == "Southeast":
        data["region_southeast"] = 1
    elif region == "Southwest":
        data["region_southwest"] = 1
    # Northeast is the reference category (all zeros) if none match
    return pd.DataFrame([data])[feature_cols]   # ensure correct column order

# ------------------------------------------------------------------
# Default info panel (before prediction)
# ------------------------------------------------------------------
if not predict_btn:
    st.info("Fill in your details in the left sidebar, then click **Predict Charges**.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size",  "1,338 records")
    col2.metric("Best Model R²", "0.8652 (Random Forest)")
    col3.metric("Avg Prediction Error", "~$4,574")

    st.markdown("---")
    st.markdown("""
### How is the prediction made?
1. You enter your personal details (age, BMI, smoker status, etc.)
2. We convert text inputs (like "Male" or "Yes") into numbers the model understands
3. The **Random Forest** model runs your numbers through 100 decision trees
4. It averages the answers of all trees to give the final prediction
5. The **cost breakdown** shows which of your inputs contributed most to the price
""")

# ------------------------------------------------------------------
# Prediction result
# ------------------------------------------------------------------
else:
    X_input = build_input(age, sex, bmi, children, smoker, region, feature_cols)

    if "Random Forest" in model_choice:
        prediction = rf_model.predict(X_input)[0]
        model_name = "Random Forest"
        model_r2   = 0.8652
    else:
        X_scaled   = scaler.transform(X_input)
        prediction = lr_model.predict(X_scaled)[0]
        model_name = "Linear Regression"
        model_r2   = 0.7836

    prediction = max(500.0, prediction)   # floor at $500

    st.markdown("---")
    st.subheader("Prediction Result")

    # Main result cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Annual Charges", f"${prediction:,.2f}")
    col2.metric("Estimated Monthly Cost",   f"${prediction/12:,.2f}")
    col3.metric(f"Model Used ({model_name})", f"R² = {model_r2:.4f}")

    # ---- Gauge chart --------------------------------------------------------
    st.markdown("---")
    st.subheader("Where Does Your Cost Fall?")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        delta={"reference": 13270, "valueformat": "$,.0f"},
        title={"text": "Predicted Annual Charges ($)"},
        number={"prefix": "$", "valueformat": ",.0f"},
        gauge={
            "axis":  {"range": [0, 60000], "tickformat": "$,.0f"},
            "bar":   {"color": "#2563eb"},
            "steps": [
                {"range": [0,     10000], "color": "#dcfce7"},
                {"range": [10000, 25000], "color": "#fef9c3"},
                {"range": [25000, 60000], "color": "#fee2e2"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 13270,   # dataset average
            },
        },
    ))
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Green = below average · Yellow = around average · Red = above average. "
        f"Black line = dataset average of $13,270. "
        f"Delta shows how you compare to the average."
    )

    # ---- Cost driver bar chart ----------------------------------------------
    st.markdown("---")
    st.subheader("Cost Drivers Breakdown")
    st.markdown(
        "This chart shows the **approximate contribution** of each factor "
        "to your total predicted charge (based on dataset averages and correlations)."
    )

    # Approximate contributions using correlation-informed weights
    df_raw = pd.read_csv("insurance.csv")
    avg_charge = df_raw["charges"].mean()

    contributions = {}
    contributions["Base (dataset avg)"] = avg_charge

    # Age contribution vs dataset average age
    age_avg   = df_raw["age"].mean()
    age_coeff = df_raw["charges"].corr(df_raw["age"]) * df_raw["charges"].std() / df_raw["age"].std()
    contributions["Age"] = age_coeff * (age - age_avg)

    # BMI contribution
    bmi_avg   = df_raw["bmi"].mean()
    bmi_coeff = df_raw["charges"].corr(df_raw["bmi"]) * df_raw["charges"].std() / df_raw["bmi"].std()
    contributions["BMI"] = bmi_coeff * (bmi - bmi_avg)

    # Smoker effect
    smoker_avg_yes = df_raw[df_raw.smoker == "yes"]["charges"].mean()
    smoker_avg_no  = df_raw[df_raw.smoker == "no"]["charges"].mean()
    contributions["Smoking"] = (smoker_avg_yes - avg_charge) if smoker == "Yes" else (smoker_avg_no - avg_charge)

    # Children
    child_avg   = df_raw["children"].mean()
    child_coeff = df_raw["charges"].corr(df_raw["children"]) * df_raw["charges"].std() / df_raw["children"].std()
    contributions["Children"] = child_coeff * (children - child_avg)

    # Region
    region_map = {"Southwest": "southwest", "Southeast": "southeast",
                  "Northwest": "northwest", "Northeast": "northeast"}
    region_avg  = df_raw[df_raw.region == region_map[region]]["charges"].mean()
    contributions["Region"] = region_avg - avg_charge

    # Sex
    sex_map = {"Male": "male", "Female": "female"}
    sex_avg = df_raw[df_raw.sex == sex_map[sex]]["charges"].mean()
    contributions["Sex"] = sex_avg - avg_charge

    factors = [(k, v) for k, v in contributions.items() if k != "Base (dataset avg)"]
    factors.sort(key=lambda x: abs(x[1]), reverse=True)
    labels = [f[0] for f in factors]
    values = [f[1] for f in factors]
    colors = ["#16a34a" if v >= 0 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Contribution to Charge ($)",
        yaxis_title="",
        height=320,
        xaxis_tickformat="$,.0f",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green bars increase your cost above average; red bars reduce it.")

    # ---- Summary text -------------------------------------------------------
    st.markdown("---")
    st.subheader("Summary")
    smoker_note = (
        "You are a **smoker**, which is the single largest cost driver — "
        "smokers pay ~4× more than non-smokers on average."
    ) if smoker == "Yes" else (
        "You are a **non-smoker**, which keeps your costs significantly lower."
    )
    bmi_note = (
        f"Your BMI of **{bmi:.1f}** is {'above' if bmi > 25 else 'within'} the healthy range (18.5–24.9)."
    )
    st.markdown(f"""
- **Estimated annual cost:** ${prediction:,.2f} &nbsp;(${prediction/12:,.2f}/month)
- {smoker_note}
- {bmi_note}
- **Model:** {model_name} with R² = {model_r2:.4f}
- **Note:** This is a statistical estimate. Actual insurance quotes depend on your specific plan, provider, and medical history.
""")
