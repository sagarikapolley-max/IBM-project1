"""
pages/1_EDA.py  —  Exploratory Data Analysis
=============================================
This page helps us UNDERSTAND the data BEFORE building models.
We answer questions like:
  - How are ages/BMIs distributed?
  - Do smokers pay more?
  - Which region is most expensive?
  - Which features correlate with charges?

Libraries used here:
  - pandas   → load and manipulate the data table
  - plotly   → interactive charts (hover to see values)
  - seaborn  → heatmap (correlation matrix)
  - matplotlib → base for seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.markdown("Visual exploration of the insurance dataset before any modelling.")

# ------------------------------------------------------------------
# Load data
# (st.cache_data caches the result so we don't reload on every click)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# ------------------------------------------------------------------
# 1. RAW DATA PREVIEW
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("1. Raw Data Preview")
st.markdown(
    f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**. "
    "Each row is one insured person."
)
st.dataframe(df.head(10), use_container_width=True)

# Summary statistics table
with st.expander("Show Summary Statistics"):
    st.dataframe(df.describe().round(2), use_container_width=True)

# ------------------------------------------------------------------
# 2. DISTRIBUTION CHARTS  (age, bmi, charges)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("2. Distribution of Numeric Features")
st.markdown(
    "A **histogram** shows how many people fall into each value bucket. "
    "The **KDE curve** (smooth line) shows the overall shape."
)

col1, col2, col3 = st.columns(3)

with col1:
    fig = px.histogram(
        df, x="age", nbins=20, title="Age Distribution",
        color_discrete_sequence=["#3b82d4"],
        labels={"age": "Age (years)", "count": "Count"},
    )
    fig.update_layout(bargap=0.05, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The dataset is fairly balanced across all ages (18–64). "
        "There is a slight spike at 18–19 (youngest bracket)."
    )

with col2:
    fig = px.histogram(
        df, x="bmi", nbins=25, title="BMI Distribution",
        color_discrete_sequence=["#7c5cd8"],
        labels={"bmi": "BMI", "count": "Count"},
    )
    fig.update_layout(bargap=0.05, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Most people have a BMI between 25–35 (overweight range). "
        "A healthy BMI is 18.5–24.9."
    )

with col3:
    fig = px.histogram(
        df, x="charges", nbins=30, title="Charges Distribution",
        color_discrete_sequence=["#10b981"],
        labels={"charges": "Charges ($)", "count": "Count"},
    )
    fig.update_layout(bargap=0.05, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Charges are right-skewed: most people pay $1k–$15k, "
        "but a small group of smokers pay $30k–$60k+."
    )

# ------------------------------------------------------------------
# 3. SMOKER vs CHARGES  (box plot)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("3. Smoker vs Charges")
st.markdown(
    "A **box plot** shows the spread: the box = middle 50% of values, "
    "the line inside = median, dots = outliers."
)

col1, col2 = st.columns([2, 1])
with col1:
    fig = px.box(
        df, x="smoker", y="charges", color="smoker",
        title="Insurance Charges: Smokers vs Non-Smokers",
        labels={"smoker": "Smoker", "charges": "Annual Charges ($)"},
        color_discrete_map={"yes": "#ef4444", "no": "#3b82d4"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    smoker_avg    = df[df.smoker == "yes"]["charges"].mean()
    nonsmoker_avg = df[df.smoker == "no"]["charges"].mean()
    st.metric("Avg Smoker Charges",     f"${smoker_avg:,.0f}")
    st.metric("Avg Non-Smoker Charges", f"${nonsmoker_avg:,.0f}")
    st.metric("Smokers Pay More By",    f"{smoker_avg/nonsmoker_avg:.1f}×")
    st.markdown(
        "> **Key insight:** Smokers pay almost **4× more** on average. "
        "Smoking is the single biggest cost driver in this dataset."
    )

# ------------------------------------------------------------------
# 4. AGE vs CHARGES  (scatter, coloured by smoker)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("4. Age vs Charges (coloured by Smoker status)")
st.markdown(
    "Each **dot** is one person. Hover to see their details. "
    "You'll notice **three bands** — smokers always sit in the top band."
)
fig = px.scatter(
    df, x="age", y="charges", color="smoker",
    title="Age vs Insurance Charges",
    labels={"age": "Age", "charges": "Charges ($)", "smoker": "Smoker"},
    color_discrete_map={"yes": "#ef4444", "no": "#3b82d4"},
    opacity=0.6, hover_data=["sex", "bmi", "region"],
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Three clear bands are visible: (1) non-smokers with low-normal BMI, "
    "(2) non-smokers with high BMI, (3) smokers — who pay significantly more regardless of age."
)

# ------------------------------------------------------------------
# 5. BMI vs CHARGES  (scatter, coloured by smoker)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("5. BMI vs Charges (coloured by Smoker status)")
st.markdown("High BMI **combined with smoking** creates the highest charges.")
fig = px.scatter(
    df, x="bmi", y="charges", color="smoker",
    title="BMI vs Insurance Charges",
    labels={"bmi": "BMI", "charges": "Charges ($)", "smoker": "Smoker"},
    color_discrete_map={"yes": "#ef4444", "no": "#3b82d4"},
    opacity=0.6, trendline="ols",
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 6. REGION vs CHARGES  (bar chart)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("6. Region vs Average Charges")
region_avg = df.groupby("region")["charges"].mean().reset_index()
region_avg.columns = ["Region", "Average Charges ($)"]
region_avg = region_avg.sort_values("Average Charges ($)", ascending=False)

fig = px.bar(
    region_avg, x="Region", y="Average Charges ($)",
    title="Average Insurance Charges by Region",
    color="Average Charges ($)", color_continuous_scale="Blues",
    text_auto="$.0f",
)
fig.update_layout(showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "The Southeast has the highest average charges — likely driven by higher smoking rates "
    "and higher average BMI in that region."
)

# ------------------------------------------------------------------
# 7. CORRELATION HEATMAP
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("7. Correlation Heatmap")
st.markdown(
    "A **correlation** value near **+1** means 'as X goes up, Y goes up'. "
    "Near **0** means no relationship. "
    "Near **−1** means 'as X goes up, Y goes down'."
)

df_num = df[["age", "bmi", "children", "charges"]].copy()
corr   = df_num.corr()

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="Blues",
    linewidths=0.5, ax=ax, vmin=-1, vmax=1,
    annot_kws={"size": 13},
)
ax.set_title("Pearson Correlation — Numeric Features", fontsize=13, pad=10)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
st.caption(
    "**Age** (0.30) and **BMI** (0.20) have the strongest positive correlations "
    "with charges. **Children** (0.07) has almost no effect."
)

# ------------------------------------------------------------------
# 8. CHILDREN vs CHARGES  (violin plot)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader("8. Number of Children vs Charges")
fig = px.violin(
    df, x="children", y="charges", box=True, color="smoker",
    title="Charges by Number of Children",
    labels={"children": "Number of Children", "charges": "Charges ($)"},
    color_discrete_map={"yes": "#ef4444", "no": "#3b82d4"},
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Having more children slightly increases charges, but the effect is small "
    "compared to smoking."
)
