# ChurnIQ — Customer Churn Prediction ML Analytics Dashboard
### *Made with ❤️ by Hashir Khan*

**ChurnIQ** is a production-grade, highly interactive Machine Learning dashboard built with **Streamlit** and **Plotly** to explore, train, evaluate, and deploy predictive models using the **IBM Telco Customer Churn** dataset.

The system features custom **glassmorphic card styling**, dynamic HSL color mappings, smooth hover animations, and rigorous ML pipelines constructed to strictly prevent data leakage.

---

## 🚀 Live Server Deployment
The dashboard is currently running live on your system:
* **Local URL:** [http://localhost:8501](http://localhost:8501)
* **Network URL:** [http://192.168.1.18:8501](http://192.168.1.18:8501)

---

## 🎨 Professional UI & Aesthetics
The visual identity of **ChurnIQ** is optimized for executive-level presentation:
* **Premium Dark Mode:** Employs slate, indigo, and violet accents paired with custom typography (Inter font family).
* **Micro-Animations:** Subtle CSS hover translations and glow effects on cards and metric highlights.
* **Streamlined Sidebar:** Compact navigation panel with descriptive icons for seamless transition across 6 distinct pages.

---

## 🕹️ System Architecture & 6 Sections

### 🏠 1. Home
* **KPI Showcase:** Dynamic metric cards reflecting **Total Customers (7,043)**, **Churn Rate (26.5%)**, **Avg Monthly Charges ($64.76)**, and **Avg Tenure (32.4 months)**.
* **High-Impact Visuals:** Donut chart showcasing customer distribution alongside stacked bar charts reflecting churn rates per contract length.
* **Core Business Case:** Highlighting why retaining existing subscribers is **5-25×** cheaper than acquiring new ones.

### 📂 2. Dataset Explorer
* **Table Search Filters:** Type columns (e.g. `gender, Contract, Churn`) to filter down massive datasets interactively.
* **Interactive Slicer:** Slider to increase/decrease the rows displayed at once (from 10 to 200).
* **Column Profiler:** Fully inspect any selected column’s stats, unique value count, missing counts, value percentage, and automatic distribution charts.

### 📈 3. EDA Dashboard
* **Univariate Tab:** Clean histograms, boxplots, and box-whiskers of numeric parameters coupled with automatic descriptive stat summaries.
* **Bivariate Tab:** Compare contract length, internet type, and payment methods vs churn rates to identify core customer friction points.
* **Correlation Thermal Heatmap:** Interactive correlation grid showcasing relationships between numeric variables.
* **Lightweight Feature Importance:** Employs a quick-scan Random Forest model to rank and display the top indicators driving churn.

### 🤖 4. Model Training
* **Leakage-Free Preprocessing:** Pipeline applies standard scaling and one-hot encoding on train splits only.
* **Class Imbalance Resolution:** Integrates **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively inside the training pipeline to balance labels without leaking into test data.
* **7 Algorithms Ready:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, and XGBoost.
* **Automatic Champions Saving:** The model hitting the highest ROC-AUC automatically serializes itself to `models/best_churn_model.pkl`.

### 🏆 5. Model Comparison
* **Metrics Leaderboard:** Interactive dataframe sorting and highlight-marking the peak performance scores.
* **radar Plot:** Polar multi-axis comparison charts showcasing models across 5 evaluation parameters.
* **ROC & Precision-Recall Overlays:** Interactive lines detailing true-positive rate gains and precision trade-offs.
* **Confusion Matrix Grid:** Side-by-side subplot heatmaps displaying raw classifications, false positives, and false negatives.

### 🎯 6. Prediction System
* **Profile Creation:** Easily dial in customer variables (contract length, payment, demographics, technical add-ons).
* **Charges Calculation:** Auto-estimates total charges mathematically via `Tenure * Monthly Charges` with a manual override badge.
* **Visual Churn Gauge:** Radial speedometer plotting prediction confidence in real-time.
* **Actionable Retention Recommendations:** Suggests personalized tactics (such as upgrading contract lengths, credit-card migration, support scheduling).

---

## 📂 Project Directory Structure

```
c:\Users\CITY COMPUTER HYD\Downloads\Project\
├── app.py                  # Main Streamlit UI Router & page definitions
├── utils.py                # Core ML helper classes, load_data, and pipelines
├── requirements.txt        # Production-pinned package requirements
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Target IBM dataset CSV
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb  # Sequenced Jupyter Notebook
│   └── 01_eda_and_preprocessing.py     # Python conversion pipeline
├── models/
│   └── best_churn_model.pkl  # Serialized top model (automatically updated)
├── visuals/                # Project screenshots, dashboard mockups, and chart graphics
└── README.md               # Extensive project documentation
```

---

## 🛠️ Local Setup & Deployment

### Prerequisites
* Python 3.13 (or 3.10+) installed.

### Installation & Execution
1. Clone or download this project directory.
2. Open PowerShell or Command Prompt inside the workspace.
3. Install pinned dependencies:
   ```bash
   & "C:/Users/CITY COMPUTER HYD/AppData/Local/Programs/Python/Python313/python.exe" -m pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   & "C:/Users/CITY COMPUTER HYD/AppData/Local/Programs/Python/Python313/python.exe" -m streamlit run app.py
   ```
5. Navigate to `http://localhost:8501` to use the dashboard!
