"""
app.py  —  Main entry point for the Streamlit app
===================================================
When you run  `streamlit run app.py`  this file opens first.
It shows a welcome/home page and Streamlit automatically discovers
the pages/ folder and builds the sidebar navigation from it.

Beginner note:
  st.XXX() functions add visual elements to the web page:
    st.title()     → big heading
    st.markdown()  → formatted text (supports **bold**, etc.)
    st.columns()   → split the page into side-by-side columns
    st.metric()    → a KPI-style number card
"""

import streamlit as st
import pandas as pd

# ------------------------------------------------------------------
# Page configuration  (must be the FIRST Streamlit call in the file)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar branding
# ------------------------------------------------------------------
st.sidebar.title("🏥 Insurance Cost Predictor")
st.sidebar.markdown(
    "Navigate using the pages above.\n\n"
    "**Dataset:** 1,338 US insurance records\n\n"
    "**Models:** Linear Regression · Random Forest"
)

# ------------------------------------------------------------------
# Home page content
# ------------------------------------------------------------------
st.title("🏥 Medical Insurance Cost Prediction")
st.markdown("### Welcome! Use the sidebar to navigate between pages.")

st.markdown("---")

# Quick dataset stats
df = pd.read_csv("insurance.csv")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",   f"{len(df):,}")
col2.metric("Avg Charges",     f"${df['charges'].mean():,.0f}")
col3.metric("Max Charges",     f"${df['charges'].max():,.0f}")
col4.metric("Smokers in Data", f"{(df['smoker']=='yes').sum()} ({(df['smoker']=='yes').mean()*100:.0f}%)")

st.markdown("---")

# Page guide
st.markdown("""
## 📌 Pages Guide

| Page | What you'll find |
|---|---|
| 📊 **EDA** | Charts exploring age, BMI, smoker status, regional differences |
| 🤖 **Model Comparison** | Side-by-side R², RMSE, MAE for Linear Regression vs Random Forest |
| 🔮 **Predictor** | Enter your details and get an instant predicted insurance cost |

> **Tip:** Select a page from the **left sidebar** to get started.
""")
