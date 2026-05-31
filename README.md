# ChurnIQ — Customer Churn Prediction ML Analytics Dashboard
### *Made with ❤️ by Hashir Khan*

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-189fdd?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
</p>

---

## 🌐 Live Demo

> **🚀 [https://customer-churn-prediction-ml-dashboard-1.streamlit.app](https://customer-churn-prediction-ml-dashboard-1.streamlit.app)**

---

## 📖 Overview

**ChurnIQ** is a production-grade, highly interactive Machine Learning dashboard built with **Streamlit** and **Plotly** to explore, train, evaluate, and deploy predictive models on the **IBM Telco Customer Churn** dataset.

Customer **churn** — when a subscriber cancels or stops using a service — is one of the most costly challenges in the telecommunications industry. Acquiring a new customer costs **5–25×** more than retaining an existing one. This dashboard provides an **end-to-end ML pipeline** to identify at-risk customers before they leave.

The system features:
- Custom **glassmorphic card styling** with a premium dark mode UI
- **Dynamic HSL color mappings** and smooth hover micro-animations
- **Rigorous ML pipelines** that strictly prevent data leakage
- **Real-time churn inference** for individual customer profiles

---

## 🎨 Professional UI & Aesthetics

- **Premium Dark Mode** — Slate, indigo, and violet accent palette with custom Inter typography
- **Micro-Animations** — Subtle CSS hover translations and glow effects on KPI cards
- **Streamlined Sidebar** — Compact navigation with descriptive icons across 6 distinct pages
- **Responsive Layouts** — Adaptive column grids for charts, tables, and controls

---

## 🕹️ System Architecture — 6 Dashboard Sections

### 🏠 1. Home
- **KPI Showcase** — Dynamic cards: Total Customers (7,043), Churn Rate (26.5%), Avg Monthly Charges ($64.76), Avg Tenure (32.4 mo)
- **Donut Chart** — Customer churn vs retained distribution
- **Bar Chart** — Churn rate by contract type
- **Business Case** — Why retaining subscribers is 5–25× cheaper than acquiring new ones
- **Project Workflow** — Step-by-step pipeline walkthrough

### 📂 2. Dataset Explorer
- **Column Search Filter** — Type column names to filter the raw data table
- **Interactive Row Slicer** — Slider to control rows displayed (10–200)
- **Data Types & Missing Values** — Side-by-side profiling panels
- **Descriptive Statistics** — Full `df.describe()` with formatted output
- **Column Profiler** — Inspect any column's value counts, unique values, and auto-distribution chart

### 📈 3. EDA Dashboard (4 Tabs)
- **Univariate** — Histograms/bar charts + dynamic insight panel per feature
- **Bivariate** — Stacked bar + churn rate charts for 8 categorical features vs churn
- **Correlation Heatmap** — Interactive Pearson correlation matrix for numeric features
- **Feature Importance** — Quick-scan Random Forest ranking of top N features (adjustable slider)

### 🤖 4. Model Training
- **7 Algorithms** — Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, XGBoost
- **Leakage-Free Pipeline** — StandardScaler + OneHotEncoder fit on **train data only**
- **SMOTE Resampling** — Applied inside training loop only (never touches test data)
- **Confusion Matrix** — Normalized heatmap with percentage annotations
- **ROC Curve** — Per-model AUC visualization after every training run
- **Auto Champion Saving** — Best ROC-AUC model serialized to `models/best_churn_model.pkl`

### 🏆 5. Model Comparison
- **Metrics Leaderboard** — Sortable, highlight-marked dataframe across Accuracy, Precision, Recall, F1, ROC-AUC
- **Champion Callout Box** — Automatically identifies and badges the best model
- **Radar Chart** — Polar multi-axis comparison across all 5 metrics
- **Combined ROC Curves** — All models on a single overlaid chart
- **Precision-Recall Curves** — Average Precision overlay for all models
- **Confusion Matrix Grid** — Side-by-side subplot heatmaps for all trained models

### 🎯 6. Prediction System
- **Customer Profile Form** — Full input form (demographics, account, phone, internet services)
- **Auto Total Charges** — Calculated as `Tenure × Monthly Charges`
- **Churn Risk Result** — Color-coded High Risk / Likely to Stay result card
- **Probability Gauge** — Radial speedometer showing churn confidence %
- **Progress Bar** — Linear churn probability indicator
- **Retention Recommendations** — Actionable tactics personalized to the prediction result
- **Input Summary** — Expandable table of all features submitted

---

## 📂 Project Directory Structure

```
customer-churn-prediction-ml-dashboard/
│
├── app.py                  # Main Streamlit UI router & all 6 page definitions
├── utils.py                # ML helpers: data loading, preprocessing, training, prediction
├── requirements.txt        # Pinned production dependencies
├── .gitignore              # Excludes __pycache__, .ipynb_checkpoints, etc.
├── README.md               # This file
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco dataset (7,043 rows × 21 cols)
│
├── models/
│   └── best_churn_model.pkl    # Auto-saved best model bundle (preprocessor + classifier)
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb  # Full Jupyter Notebook with EDA, tuning & modeling
│   └── 01_eda_and_preprocessing.py     # Python script version of the notebook
│
└── visuals/
    └── *.png               # Dashboard screenshots and chart exports
```

---

## 🛠️ Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| **UI** | Streamlit 1.x | Multi-page dashboard framework |
| **Visualization** | Plotly, Plotly Express | Interactive charts & gauges |
| **Data** | Pandas, NumPy | Data wrangling & numerical ops |
| **ML** | Scikit-learn | Preprocessing, modeling, metrics |
| **Boosting** | XGBoost | Gradient boosting classifier |
| **Resampling** | Imbalanced-learn | SMOTE for class imbalance |
| **Serialization** | Joblib | Model bundle save/load |
| **Acceleration** | PyArrow | Fast dataframe serialization |

---

## ⚙️ Local Setup & Deployment

### Prerequisites
- Python **3.10+** (tested on 3.13)
- `pip` or `pip3`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/gh-hashir/customer-churn-prediction-ml-dashboard.git
cd customer-churn-prediction-ml-dashboard

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

### Access
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 📊 Dataset

**IBM Telco Customer Churn** — [Kaggle Source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Attribute | Value |
|---|---|
| Rows | 7,043 customers |
| Features | 21 columns |
| Target | `Churn` (Yes / No → 1 / 0) |
| Churn Rate | ~26.5% |
| Missing Values | 11 (TotalCharges — filled with 0) |

**Feature Categories:**
- **Demographics** — gender, SeniorCitizen, Partner, Dependents
- **Account** — tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Phone Services** — PhoneService, MultipleLines
- **Internet Services** — InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

---

## 🤖 ML Pipeline Design

```
Raw CSV
  ↓
Stratified 80/20 Train-Test Split
  ↓
ColumnTransformer (fit on TRAIN only)
  ├─ StandardScaler → [SeniorCitizen, tenure, MonthlyCharges, TotalCharges]
  └─ OneHotEncoder  → [gender, Partner, Dependents, ..., PaymentMethod]
  ↓
SMOTE (applied to processed TRAIN only)
  ↓
Classifier Training
  ↓
Evaluation on UNSEEN TEST SET
  ↓
Best Model → models/best_churn_model.pkl
```

**No data leakage** — all transformations are fit exclusively on training data.

---

## 📈 Model Results (Sample)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | ~0.81 | ~0.67 | ~0.52 | ~0.58 | ~0.85 |
| Random Forest | ~0.80 | ~0.65 | ~0.51 | ~0.57 | ~0.84 |
| XGBoost | ~0.80 | ~0.64 | ~0.53 | ~0.58 | ~0.84 |
| Logistic Regression | ~0.79 | ~0.63 | ~0.56 | ~0.59 | ~0.84 |

> *Results vary slightly per run due to SMOTE randomness.*

---

## 🔗 Links

- **Live App** → [https://customer-churn-prediction-ml-dashboard-1.streamlit.app](https://customer-churn-prediction-ml-dashboard-1.streamlit.app)
- **GitHub Repo** → [https://github.com/gh-hashir/customer-churn-prediction-ml-dashboard](https://github.com/gh-hashir/customer-churn-prediction-ml-dashboard)
- **Dataset** → [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 👤 Author

**Hashir Khan**
- GitHub: [@gh-hashir](https://github.com/gh-hashir)

---

<p align="center">Made with ❤️ using Streamlit · IBM Telco Customer Churn Dataset</p>
