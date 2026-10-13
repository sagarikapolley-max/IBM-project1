# 🏥 Medical Insurance Cost Prediction

A full end-to-end machine learning project that predicts medical insurance charges based on personal attributes, built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 📌 Project Overview

Insurance companies use factors like age, BMI, smoking status, and region to estimate a person's annual medical expenses. This project builds two ML models to predict those charges and serves them through an interactive web app.

**Dataset:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**Records:** 1,338 rows × 7 columns

---

## 🗂️ Project Structure

```
insurance_project/
│
├── insurance.csv              # Raw dataset
├── app.py                     # Streamlit home page
├── train_model.py             # Trains & saves ML models
├── requirements.txt           # Python dependencies
│
├── models/                    # Saved model files (auto-generated)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
│
└── pages/
    ├── 1_EDA.py               # Exploratory Data Analysis
    ├── 2_Model_Comparison.py  # LR vs Random Forest metrics
    └── 3_Predictor.py         # Interactive prediction tool
```

---

## 📊 Dataset Features

| Column | Type | Description |
|--------|------|-------------|
| `age` | Integer | Age of the primary beneficiary |
| `sex` | Categorical | Gender (male / female) |
| `bmi` | Float | Body Mass Index |
| `children` | Integer | Number of dependents |
| `smoker` | Categorical | Smoking status (yes / no) |
| `region` | Categorical | US residential region |
| `charges` | Float | Annual medical insurance cost (**target**) |

---

## 🤖 Models & Results

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.7836 | $5,796 | $4,181 |
| **Random Forest** ✅ | **0.8652** | **$4,574** | **$2,539** |

**Random Forest** is the recommended model — it handles non-linear relationships (e.g., smoker × BMI interaction) that Linear Regression cannot.

---

## 🖥️ App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Dataset overview with KPI metrics |
| 📊 EDA | 8 interactive charts: distributions, smoker vs charges, scatter plots, heatmap |
| 🤖 Model Comparison | R², RMSE, MAE side-by-side, actual vs predicted, residuals, feature importance |
| 🔮 Predictor | Enter age, BMI, smoker status etc. → get instant predicted insurance cost |

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` / `seaborn` | Static charts and heatmaps |
| `plotly` | Interactive charts in Streamlit |
| `scikit-learn` | Model training, evaluation, scaling |
| `statsmodels` | OLS trendlines in scatter plots |
| `streamlit` | Web app frontend |
| `joblib` | Saving and loading trained models |

---

## 🔑 Key Findings

- **Smoking** is the single biggest cost driver — smokers pay ~4× more than non-smokers
- **Age** has a moderate positive correlation (R = 0.30) with charges
- **BMI** matters most when combined with smoking
- **Region** and **number of children** have relatively small effects

---

## 👩‍💻 Author

**Sagarika Polley**  
IBM Project — Medical Insurance Cost Prediction
