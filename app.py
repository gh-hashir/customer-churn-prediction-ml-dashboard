# =============================================================================
# app.py — Customer Churn Prediction Dashboard
# =============================================================================
# Multi-page Streamlit application with 6 sections:
#   1. Home            — KPI cards, project overview
#   2. Dataset Explorer — Raw data, column profiling
#   3. EDA Dashboard   — Univariate, Bivariate, Correlation, Feature Importance
#   4. Model Training  — Interactive training with pipeline + SMOTE
#   5. Model Comparison— Leaderboard, ROC, PR curves, CM grid, Champion callout
#   6. Prediction      — Single-customer churn risk form
# =============================================================================

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Import project utilities
import utils

# =============================================================================
# PAGE CONFIG & GLOBAL STYLE
# =============================================================================

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for premium look
st.markdown(
    """
    <style>
    /* ── Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        padding: 0.35rem 0;
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 0.3rem 0;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94a3b8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .kpi-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }

    /* ── Section headers ── */
    .section-header {
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .section-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* ── Metric pills ── */
    .metric-pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-pill .val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #34d399;
    }
    .metric-pill .lbl {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Champion badge ── */
    .champion-box {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 2px solid #34d399;
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-top: 1rem;
    }
    .champion-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #34d399;
    }

    /* ── Prediction result ── */
    .result-high {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .result-prob {
        font-size: 1.1rem;
        color: #e2e8f0;
    }

    /* ── Workflow steps ── */
    .step-card {
        background: #1e293b;
        border-left: 4px solid #0ea5e9;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .step-num {
        font-size: 0.72rem;
        color: #0ea5e9;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .step-title {
        font-size: 0.98rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .step-desc {
        font-size: 0.82rem;
        color: #94a3b8;
    }

    /* ── Tech badge ── */
    .tech-badge {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.8rem;
        color: #7dd3fc;
        margin: 0.2rem;
        font-weight: 500;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #1e293b;
        margin-top: 3rem;
    }

    /* ── Divider ── */
    .custom-hr {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 2rem 0;
    }

    /* Override Streamlit's default dark/light mode backgrounds */
    .stApp {
        background-color: #020817;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <div style='font-size:2.2rem'>📊</div>
            <div style='font-size:1rem; font-weight:700; color:#38bdf8; margin-top:0.3rem;'>
                ChurnIQ
            </div>
            <div style='font-size:0.72rem; color:#64748b; letter-spacing:0.05em;'>
                ML ANALYTICS DASHBOARD
            </div>
        </div>
        <hr style='border-color:#1e293b; margin:1rem 0;'/>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Home",
            "📂  Dataset Explorer",
            "📈  EDA Dashboard",
            "🤖  Model Training",
            "🏆  Model Comparison",
            "🎯  Prediction System",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <hr style='border-color:#1e293b; margin:1rem 0;'/>
        <div style='font-size:0.72rem; color:#475569; text-align:center; line-height:1.4;'>
            Made with ❤️ by Hashir Khan<br/>
            Telco Customer Churn · v1.0
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# SHARED DATA LOAD
# =============================================================================

df = utils.load_data()

# ── Session state init ─────────────────────────────────────────────────────────
if "trained_models" not in st.session_state:
    st.session_state["trained_models"] = {}   # {model_name: results_dict}
if "trained_bundles" not in st.session_state:
    st.session_state["trained_bundles"] = {}  # {model_name: bundle_dict}


# =============================================================================
# HELPER: pretty section header
# =============================================================================

def section_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<hr class="custom-hr"/>', unsafe_allow_html=True)


# =============================================================================
# PLOTLY THEME DEFAULTS
# =============================================================================

PLOTLY_THEME = "plotly_dark"
PALETTE = px.colors.qualitative.Set2
COLOR_CHURN = "#ef4444"
COLOR_STAY = "#22c55e"
COLOR_PRIMARY = "#0ea5e9"


def apply_chart_style(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent dark styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#e2e8f0", size=12),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155")
    return fig


# =============================================================================
# SECTION 1 — HOME
# =============================================================================

def page_home():
    # ── Hero ──
    st.markdown(
        """
        <div style='text-align:center; padding: 2.5rem 0 1rem;'>
            <div style='font-size:0.8rem; font-weight:600; color:#0ea5e9;
                        letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.5rem;'>
                Enterprise ML Analytics
            </div>
            <div style='font-size:2.8rem; font-weight:800;
                        background: linear-gradient(90deg,#38bdf8,#818cf8,#f472b6);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        line-height:1.2;'>
                Customer Churn Prediction<br/>& Comparative ML Analysis
            </div>
            <div style='color:#94a3b8; font-size:1rem; margin-top:0.8rem; max-width:600px;
                        margin-left:auto; margin-right:auto;'>
                Identify at-risk customers before they leave — powered by
                ensemble machine learning, interactive EDA, and real-time inference.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    divider()

    # ── KPI Cards ──
    churn_rate = df["Churn"].mean() * 100
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "👥", f"{len(df):,}", "Total Customers"),
        (c2, "⚠️", f"{churn_rate:.1f}%", "Churn Rate"),
        (c3, "💰", f"${avg_monthly:.2f}", "Avg Monthly Charges"),
        (c4, "📅", f"{avg_tenure:.1f} mo", "Avg Tenure"),
    ]
    for col, icon, value, label in kpis:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    divider()

    # ── Business Problem ──
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("### 🎯 Business Problem")
        st.markdown(
            """
            Customer **churn** — when a subscriber cancels or stops using a service — is one of
            the most costly challenges in the telecommunications industry. Acquiring a new customer
            costs **5–25×** more than retaining an existing one.

            This dashboard provides an **end-to-end ML pipeline** to:
            - Explore the patterns behind customer churn
            - Train and compare multiple classifiers
            - Predict individual customer churn risk in real time
            - Guide targeted retention campaigns

            **Dataset:** IBM Telco Customer Churn — **7,043 customers**, **21 features** including
            demographics, account info, and service subscriptions.
            """
        )

        divider()

        # ── Project Workflow ──
        st.markdown("### 🔄 Project Workflow")
        steps = [
            ("Step 1", "Exploratory Data Analysis",
             "Understand distributions, correlations, and churn drivers."),
            ("Step 2", "Data Preprocessing",
             "Handle missing values, encode categoricals, scale numerics."),
            ("Step 3", "Model Training & SMOTE",
             "Train 6 classifiers with class-imbalance correction."),
            ("Step 4", "Model Comparison",
             "Evaluate ROC-AUC, F1, Precision-Recall across all models."),
            ("Step 5", "Deployment & Inference",
             "Real-time prediction for individual customers."),
        ]
        for num, title, desc in steps:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### 📊 Churn Distribution")
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map({0: "Retained", 1: "Churned"})

        fig_pie = px.pie(
            churn_counts,
            values="Count",
            names="Label",
            color="Label",
            color_discrete_map={"Churned": COLOR_CHURN, "Retained": COLOR_STAY},
            hole=0.55,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie = apply_chart_style(fig_pie, height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

        divider()

        st.markdown("### 🛠 Technology Stack")
        techs = [
            "Streamlit", "Plotly", "Pandas", "NumPy",
            "Scikit-learn", "XGBoost", "Imbalanced-learn", "Joblib",
        ]
        badges = "".join(f'<span class="tech-badge">{t}</span>' for t in techs)
        st.markdown(badges, unsafe_allow_html=True)

        divider()

        # Quick churn insight mini-chart
        st.markdown("### 💡 Churn by Contract Type")
        contract_churn = (
            df.groupby("Contract")["Churn"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"Churn": "Churn Rate (%)"})
        )
        fig_bar = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate (%)",
            color="Contract",
            color_discrete_sequence=[COLOR_CHURN, "#f59e0b", COLOR_STAY],
            text_auto=".1f",
        )
        fig_bar = apply_chart_style(fig_bar, height=260)
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Footer ──
    st.markdown(
        """
        <div class="footer">
            Made with ❤️ by Hashir Khan using Streamlit · IBM Telco Customer Churn Dataset
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 2 — DATASET EXPLORER
# =============================================================================

def page_dataset_explorer():
    section_header(
        "Dataset Explorer",
        "Browse, filter, and profile the raw Telco Customer Churn dataset.",
    )

    # ── Summary stats row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))
    with c4:
        st.metric("Duplicates", int(df.duplicated().sum()))

    divider()

    # ── Raw Data Table ──
    st.markdown("#### 📋 Raw Dataset")
    col_search, col_slider = st.columns([3, 1])
    with col_search:
        search = st.text_input(
            "Search columns (comma-separated column names, or leave blank)",
            placeholder="e.g. gender, Contract, Churn",
            key="ds_search",
        )
    with col_slider:
        n_rows = st.slider("Rows to show", 10, 200, 50, 10, key="ds_rows")

    display_df = df.copy()
    if search.strip():
        cols = [c.strip() for c in search.split(",") if c.strip() in df.columns]
        if cols:
            display_df = display_df[cols]
        else:
            st.warning("No matching column names found.")

    st.dataframe(
        display_df.head(n_rows),
        use_container_width=True,
        height=380,
    )

    divider()

    # ── Data Types & Missing Values ──
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### 🗂 Data Types")
        dtypes_df = pd.DataFrame({
            "Column": df.dtypes.index,
            "Type": df.dtypes.astype(str).values,
        })
        st.dataframe(dtypes_df, use_container_width=True, height=320, hide_index=True)

    with right:
        st.markdown("#### 🔍 Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing["% Missing"] = (missing["Missing"] / len(df) * 100).round(2)
        st.dataframe(missing, use_container_width=True, height=320, hide_index=True)

    divider()

    # ── Descriptive Statistics ──
    with st.expander("📊 Descriptive Statistics (df.describe())", expanded=False):
        st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

    divider()

    # ── Column Profiling ──
    st.markdown("#### 🔬 Column Profiler")
    selected_col = st.selectbox(
        "Select a column to profile",
        options=df.columns.tolist(),
        index=df.columns.tolist().index("Churn"),
        key="profile_col",
    )

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown(f"**Type:** `{df[selected_col].dtype}`")
        st.markdown(f"**Unique values:** {df[selected_col].nunique()}")
        st.markdown(f"**Missing:** {df[selected_col].isnull().sum()}")
        st.markdown("**Value Counts:**")
        vc = df[selected_col].value_counts().reset_index()
        vc.columns = [selected_col, "Count"]
        vc["% "] = (vc["Count"] / len(df) * 100).round(1)
        st.dataframe(vc, use_container_width=True, hide_index=True, height=280)

    with col_right:
        if not pd.api.types.is_numeric_dtype(df[selected_col]):
            vc_plot = df[selected_col].value_counts().reset_index()
            vc_plot.columns = [selected_col, "Count"]
            fig = px.bar(
                vc_plot,
                x=selected_col,
                y="Count",
                color=selected_col,
                color_discrete_sequence=PALETTE,
                title=f"Distribution of {selected_col}",
            )
        else:
            fig = px.histogram(
                df,
                x=selected_col,
                nbins=40,
                color_discrete_sequence=[COLOR_PRIMARY],
                title=f"Distribution of {selected_col}",
            )
        fig = apply_chart_style(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SECTION 3 — EDA DASHBOARD
# =============================================================================

def page_eda():
    section_header(
        "EDA Dashboard",
        "Interactive exploratory data analysis with dynamic insights.",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Univariate", "🔗 Bivariate", "🌡️ Correlation", "⭐ Feature Importance"]
    )

    # ── TAB 1 : Univariate ────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Univariate Analysis")
        uni_col = st.selectbox(
            "Select feature",
            options=["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
                     "gender", "Contract", "InternetService", "PaymentMethod"],
            key="uni_col",
        )

        col_l, col_r = st.columns([3, 1], gap="large")

        with col_l:
            if not pd.api.types.is_numeric_dtype(df[uni_col]):
                vc = df[uni_col].value_counts().reset_index()
                vc.columns = [uni_col, "Count"]
                fig = px.bar(
                    vc, x=uni_col, y="Count",
                    color=uni_col,
                    color_discrete_sequence=PALETTE,
                    title=f"Count Distribution — {uni_col}",
                )
            else:
                fig = px.histogram(
                    df, x=uni_col, nbins=50,
                    marginal="box",
                    color_discrete_sequence=[COLOR_PRIMARY],
                    title=f"Distribution — {uni_col}",
                )
            fig = apply_chart_style(fig, height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("##### 📌 Quick Stats")
            col_data = df[uni_col]
            if pd.api.types.is_numeric_dtype(col_data):
                stats = {
                    "Mean": f"{col_data.mean():.2f}",
                    "Median": f"{col_data.median():.2f}",
                    "Std Dev": f"{col_data.std():.2f}",
                    "Min": f"{col_data.min():.2f}",
                    "Max": f"{col_data.max():.2f}",
                    "Skewness": f"{col_data.skew():.2f}",
                }
                for k, v in stats.items():
                    st.markdown(
                        f"<div class='metric-pill'><div class='val'>{v}</div>"
                        f"<div class='lbl'>{k}</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(f"**Unique:** {col_data.nunique()}")
                st.markdown(f"**Top value:** `{col_data.mode()[0]}`")
                st.markdown(f"**Top %:** {col_data.value_counts(normalize=True).iloc[0]*100:.1f}%")

        # Dynamic insight
        with st.expander("💡 Dynamic Insight", expanded=True):
            _univariate_insight(uni_col)

    # ── TAB 2 : Bivariate ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Bivariate Analysis — Churn vs Feature")
        bivar_options = {
            "Contract": "Contract Type",
            "PaymentMethod": "Payment Method",
            "InternetService": "Internet Service",
            "gender": "Gender",
            "SeniorCitizen": "Senior Citizen",
            "Partner": "Partner",
            "Dependents": "Dependents",
            "PaperlessBilling": "Paperless Billing",
        }
        bivar_col = st.selectbox(
            "Select comparison feature",
            options=list(bivar_options.keys()),
            format_func=lambda x: bivar_options[x],
            key="bivar_col",
        )

        churn_label_map = {0: "Retained", 1: "Churned"}
        plot_df = df.copy()
        plot_df["Churn Label"] = plot_df["Churn"].map(churn_label_map)

        if bivar_col == "SeniorCitizen":
            plot_df[bivar_col] = plot_df[bivar_col].map({0: "No", 1: "Yes"})

        # Stacked bar
        bivar_agg = (
            plot_df.groupby([bivar_col, "Churn Label"])
            .size()
            .reset_index(name="Count")
        )
        fig_stack = px.bar(
            bivar_agg,
            x=bivar_col,
            y="Count",
            color="Churn Label",
            barmode="stack",
            color_discrete_map={"Churned": COLOR_CHURN, "Retained": COLOR_STAY},
            title=f"Churn Distribution by {bivar_options[bivar_col]}",
        )
        fig_stack = apply_chart_style(fig_stack)
        st.plotly_chart(fig_stack, use_container_width=True)

        # Churn rate per category
        rate_df = (
            df.groupby(bivar_col)["Churn"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"Churn": "Churn Rate (%)"})
        )
        if bivar_col == "SeniorCitizen":
            rate_df[bivar_col] = rate_df[bivar_col].map({0: "No", 1: "Yes"})

        fig_rate = px.bar(
            rate_df,
            x=bivar_col,
            y="Churn Rate (%)",
            text_auto=".1f",
            color="Churn Rate (%)",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            title=f"Churn Rate (%) by {bivar_options[bivar_col]}",
        )
        fig_rate.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_rate = apply_chart_style(fig_rate)
        st.plotly_chart(fig_rate, use_container_width=True)

        with st.expander("💡 Insight Summary", expanded=True):
            _bivariate_insight(bivar_col, rate_df)

    # ── TAB 3 : Correlation ───────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Correlation Heatmap (Numerical Features)")
        num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        corr = df[num_cols].corr().round(3)

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu",
                reversescale=True,
                zmid=0,
                text=corr.values,
                texttemplate="%{text:.2f}",
                hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
            )
        )
        fig_heat.update_layout(title="Pearson Correlation Matrix")
        fig_heat = apply_chart_style(fig_heat, height=450)
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("💡 Correlation Insights"):
            st.markdown(
                """
                - **Tenure ↔ TotalCharges** (strong positive): Longer customers accumulate higher charges.
                - **Tenure ↔ Churn** (negative): Newer customers churn significantly more.
                - **MonthlyCharges ↔ Churn** (positive): Higher monthly fees correlate with churn.
                - **MonthlyCharges ↔ TotalCharges** (positive): Linked through tenure.
                """
            )

    # ── TAB 4 : Feature Importance ────────────────────────────────────────────
    with tab4:
        st.markdown("#### Top Feature Importances — Random Forest (Quick Scan)")
        st.info(
            "Training a lightweight Random Forest on label-encoded data to rank features. "
            "This is for EDA purposes only and does not affect model training in Section 4.",
            icon="ℹ️",
        )

        with st.spinner("Computing feature importances..."):
            fi = _compute_quick_feature_importance(df)

        top_n = st.slider("Top N features", 5, 20, 10, key="fi_n")
        fi_top = fi.head(top_n).reset_index()
        fi_top.columns = ["Feature", "Importance"]

        fig_fi = px.bar(
            fi_top,
            y="Feature",
            x="Importance",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#0ea5e9", "#8b5cf6", "#f472b6"],
            title=f"Top {top_n} Feature Importances",
        )
        fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
        fig_fi = apply_chart_style(fig_fi, height=420)
        st.plotly_chart(fig_fi, use_container_width=True)


@st.cache_data(show_spinner=False)
def _compute_quick_feature_importance(df: pd.DataFrame):
    """Compute RF feature importances using label encoding (EDA only)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    df_enc = df.copy()
    for col in df_enc.select_dtypes(exclude=["number"]).columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
    df_enc = df_enc.drop(columns=["customerID"], errors="ignore")

    X_t = df_enc.drop(columns=["Churn"])
    y_t = df_enc["Churn"]
    rf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    rf.fit(X_t, y_t)
    return pd.Series(rf.feature_importances_, index=X_t.columns).sort_values(ascending=False)


def _univariate_insight(col: str):
    insights = {
        "SeniorCitizen": "Only ~16% of customers are senior citizens, making this a minority group. "
                         "Senior citizens churn at a higher rate (~42%) compared to non-seniors (~24%).",
        "tenure": "Tenure is bimodal — many short-term (1-2 months) and long-term (>60 months) customers. "
                  "Customers with < 12 months tenure churn at 3× the rate of those with > 60 months.",
        "MonthlyCharges": "Monthly charges are roughly uniform between $20–$110 with a spike at the lower end. "
                          "Customers paying > $65/month have significantly higher churn rates.",
        "TotalCharges": "Total charges is right-skewed — most customers have low totals (new customers). "
                        "High TotalCharges generally indicates long tenure and lower churn risk.",
        "gender": "Gender distribution is nearly equal (49.5% Female / 50.5% Male). "
                  "Gender has minimal predictive power for churn.",
        "Contract": "Month-to-month contracts represent 55% of customers and have a 43% churn rate. "
                    "Two-year contracts have only 3% churn — contract type is a top churn predictor.",
        "InternetService": "Fiber optic customers churn at ~42%, nearly double DSL (~19%). "
                           "No internet service customers are highly retained (~7% churn).",
        "PaymentMethod": "Electronic check users churn at 45%, highest among all payment methods. "
                         "Auto-payment methods (credit card, bank transfer) have ~17% churn.",
    }
    msg = insights.get(col, f"Select a column from the list to see a dynamic insight for **{col}**.")
    st.markdown(msg)


def _bivariate_insight(col: str, rate_df: pd.DataFrame):
    highest = rate_df.loc[rate_df["Churn Rate (%)"].idxmax()]
    lowest = rate_df.loc[rate_df["Churn Rate (%)"].idxmin()]
    st.markdown(
        f"- **Highest churn**: `{highest[col]}` at **{highest['Churn Rate (%)']:.1f}%**\n"
        f"- **Lowest churn**: `{lowest[col]}` at **{lowest['Churn Rate (%)']:.1f}%**\n"
        f"- The gap of **{highest['Churn Rate (%)'] - lowest['Churn Rate (%)']:.1f}pp** "
        f"between categories suggests `{col}` is a **meaningful churn predictor**."
    )


# =============================================================================
# SECTION 4 — MODEL TRAINING
# =============================================================================

def page_model_training():
    section_header(
        "Model Training",
        "Select a classifier, train it, and inspect its performance metrics.",
    )

    available_models = [
        "Logistic Regression", "Decision Tree", "Random Forest",
        "KNN", "SVM", "Gradient Boosting", "XGBoost",
    ]

    col_ctrl, col_info = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("#### ⚙️ Training Controls")
        model_name = st.selectbox(
            "Select Model",
            options=available_models,
            key="train_model_select",
        )

        with st.expander("ℹ️ Pipeline Details"):
            st.markdown(
                """
                **No data leakage guarantee:**
                1. Stratified 80/20 train-test split
                2. `StandardScaler` + `OneHotEncoder` fit on **train only**
                3. **SMOTE** applied to processed **train set only**
                4. Model trained on resampled data
                5. Evaluation on **unseen test set**
                """
            )

        train_btn = st.button(
            f"🚀 Train {model_name}",
            type="primary",
            use_container_width=True,
            key="train_btn",
        )

        if st.session_state["trained_models"]:
            st.markdown("---")
            st.markdown("**✅ Trained so far:**")
            for m in st.session_state["trained_models"]:
                auc = st.session_state["trained_models"][m].get("ROC-AUC", 0)
                st.markdown(f"- `{m}` — AUC {auc:.3f}")

    with col_info:
        if train_btn:
            with st.spinner(f"Training {model_name}… This may take a moment."):
                try:
                    results, bundle = utils.train_model(model_name, df)
                    st.session_state["trained_models"][model_name] = results
                    st.session_state["trained_bundles"][model_name] = bundle

                    # Auto-save as best model if it has the highest AUC
                    best_auc = max(
                        (r.get("ROC-AUC", 0) or 0)
                        for r in st.session_state["trained_models"].values()
                    )
                    if (results.get("ROC-AUC") or 0) >= best_auc:
                        utils.save_model(bundle)

                    st.success(f"✅ **{model_name}** trained successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Training failed: {e}", icon="❌")
                    return

            _render_training_results(model_name, results)

        elif model_name in st.session_state["trained_models"]:
            st.info(
                f"`{model_name}` has already been trained. Results below:",
                icon="📊",
            )
            _render_training_results(
                model_name, st.session_state["trained_models"][model_name]
            )
        else:
            st.markdown(
                """
                <div style='background:#0f172a; border:1px dashed #334155;
                            border-radius:12px; padding:3rem; text-align:center;
                            color:#64748b; margin-top:1rem;'>
                    <div style='font-size:3rem;'>🤖</div>
                    <div style='font-size:1rem; margin-top:0.5rem;'>
                        Select a model and click <strong>Train</strong> to begin
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_training_results(model_name: str, results: dict):
    """Display metrics + confusion matrix for a single trained model."""
    st.markdown(f"### 📊 {model_name} — Results")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        val = results.get(metric)
        display_val = f"{val:.4f}" if val is not None else "N/A"
        with cols[i]:
            st.markdown(
                f"""
                <div class="metric-pill">
                    <div class="val">{display_val}</div>
                    <div class="lbl">{metric}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Confusion matrix
    cm = np.array(results["confusion_matrix"])
    labels = ["Retained", "Churned"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm_pct,
            x=[f"Pred {l}" for l in labels],
            y=[f"Actual {l}" for l in labels],
            colorscale="Blues",
            text=[[f"{cm[i][j]}<br>{cm_pct[i][j]:.1f}%" for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
            showscale=True,
        )
    )
    fig_cm.update_layout(title=f"Confusion Matrix — {model_name}")
    fig_cm = apply_chart_style(fig_cm, height=360)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve (if available)
    if results.get("roc_data"):
        roc = results["roc_data"]
        auc_val = results.get("ROC-AUC", 0)
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=roc["fpr"], y=roc["tpr"],
                mode="lines",
                name=f"{model_name} (AUC={auc_val:.3f})",
                line=dict(color=COLOR_PRIMARY, width=2.5),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="#64748b", dash="dash"),
            )
        )
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        fig_roc = apply_chart_style(fig_roc, height=380)
        st.plotly_chart(fig_roc, use_container_width=True)


# =============================================================================
# SECTION 5 — MODEL COMPARISON
# =============================================================================

def page_model_comparison():
    section_header(
        "Model Comparison",
        "Compare all trained models across every performance dimension.",
    )

    trained = st.session_state["trained_models"]

    if not trained:
        st.warning(
            "No models trained yet. Go to **🤖 Model Training** and train at least one model.",
            icon="⚠️",
        )
        return

    # Build leaderboard DataFrame
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rows = []
    for name, res in trained.items():
        row = {"Model": name}
        for m in metrics:
            row[m] = round(res.get(m) or 0, 4)
        rows.append(row)
    lb_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)

    # Identify champion
    champion = lb_df.iloc[0]["Model"]
    champion_auc = lb_df.iloc[0]["ROC-AUC"]

    # Champion callout
    st.markdown(
        f"""
        <div class="champion-box">
            <div class="champion-title">🏆 Champion Model — {champion}</div>
            <div style='color:#a7f3d0; margin-top:0.4rem;'>
                Highest ROC-AUC: <strong>{champion_auc:.4f}</strong> ·
                Saved to <code>models/best_churn_model.pkl</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Leaderboard ──
    st.markdown("#### 📋 Metrics Leaderboard")
    st.dataframe(
        lb_df.reset_index(drop=True).style.highlight_max(
            subset=metrics, color="#064e3b"
        ).format({m: "{:.4f}" for m in metrics}),
        use_container_width=True,
        height=min(60 + len(lb_df) * 36, 420),
        hide_index=True,
    )

    divider()

    # ── Accuracy vs F1 grouped bar ──
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("#### 📊 Accuracy vs F1-Score")
        melt_df = lb_df[["Model", "Accuracy", "F1-Score"]].melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )
        fig_bar = px.bar(
            melt_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=[COLOR_PRIMARY, "#8b5cf6"],
            text_auto=".3f",
        )
        fig_bar.update_layout(xaxis_tickangle=-30)
        fig_bar = apply_chart_style(fig_bar, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown("#### 📊 All Metrics Radar")
        fig_radar = go.Figure()
        for _, row in lb_df.iterrows():
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill="toself",
                    name=row["Model"],
                    opacity=0.7,
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Performance Radar Chart",
        )
        fig_radar = apply_chart_style(fig_radar, height=380)
        st.plotly_chart(fig_radar, use_container_width=True)

    divider()

    # ── Multi-model ROC ──
    st.markdown("#### 📈 Combined ROC Curves")
    fig_roc = go.Figure()
    for name, res in trained.items():
        if res.get("roc_data"):
            roc = res["roc_data"]
            auc_val = res.get("ROC-AUC", 0)
            fig_roc.add_trace(
                go.Scatter(
                    x=roc["fpr"], y=roc["tpr"],
                    mode="lines",
                    name=f"{name} (AUC={auc_val:.3f})",
                    line=dict(width=2),
                )
            )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(color="#64748b", dash="dash"),
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curves — All Models",
    )
    fig_roc = apply_chart_style(fig_roc, height=430)
    st.plotly_chart(fig_roc, use_container_width=True)

    divider()

    # ── Precision-Recall curves ──
    st.markdown("#### 📉 Precision-Recall Curves")
    fig_pr = go.Figure()
    for name, res in trained.items():
        if res.get("pr_data"):
            pr = res["pr_data"]
            fig_pr.add_trace(
                go.Scatter(
                    x=pr["recall"], y=pr["precision"],
                    mode="lines",
                    name=f"{name} (AP={pr['ap']:.3f})",
                    line=dict(width=2),
                )
            )
    fig_pr.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        title="Precision-Recall Curves — All Models",
    )
    fig_pr = apply_chart_style(fig_pr, height=430)
    st.plotly_chart(fig_pr, use_container_width=True)

    divider()

    # ── Confusion Matrix Grid ──
    st.markdown("#### 🔲 Confusion Matrix Grid")
    n = len(trained)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    labels = ["Retained", "Churned"]

    fig_grid = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=list(trained.keys()),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, (name, res) in enumerate(trained.items()):
        row = idx // ncols + 1
        col = idx % ncols + 1
        cm = np.array(res["confusion_matrix"])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        fig_grid.add_trace(
            go.Heatmap(
                z=cm_pct,
                x=[f"Pred {l}" for l in labels],
                y=[f"Actual {l}" for l in labels],
                colorscale="Blues",
                text=[[f"{cm[i][j]}<br>{cm_pct[i][j]:.1f}%" for j in range(2)] for i in range(2)],
                texttemplate="%{text}",
                showscale=False,
            ),
            row=row, col=col,
        )

    fig_grid.update_layout(
        height=nrows * 280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#e2e8f0", size=11),
    )
    st.plotly_chart(fig_grid, use_container_width=True)


# =============================================================================
# SECTION 6 — PREDICTION SYSTEM
# =============================================================================

def page_prediction():
    section_header(
        "Prediction System",
        "Enter customer details and get real-time churn risk assessment.",
    )

    # ── Load model ──
    # Try disk first, then session state
    bundle = utils.load_model()
    if bundle is None and st.session_state["trained_bundles"]:
        # Pick the bundle with best AUC from session
        best_name = max(
            st.session_state["trained_models"],
            key=lambda n: st.session_state["trained_models"][n].get("ROC-AUC") or 0,
        )
        bundle = st.session_state["trained_bundles"][best_name]
        st.info(
            f"Using in-session model: **{best_name}**. "
            "Train models in Section 4 to save permanently.",
            icon="ℹ️",
        )
    elif bundle is None:
        st.warning(
            "No trained model found. Please go to **🤖 Model Training** and train at least one model.",
            icon="⚠️",
        )
        return

    st.markdown("#### 📝 Customer Profile Form")

    # ── Form ──
    with st.form("prediction_form"):
        st.markdown("**Demographics**")
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", ["Male", "Female"], key="f_gender")
        senior = d2.selectbox("Senior Citizen", ["No", "Yes"], key="f_senior")
        partner = d3.selectbox("Partner", ["Yes", "No"], key="f_partner")
        dependents = d4.selectbox("Dependents", ["Yes", "No"], key="f_dep")

        divider()
        st.markdown("**Account Information**")
        a1, a2, a3 = st.columns(3)
        tenure = a1.slider("Tenure (months)", 0, 72, 12, key="f_tenure")
        contract = a2.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="f_contract")
        paperless = a3.selectbox("Paperless Billing", ["Yes", "No"], key="f_paper")

        a4, a5 = st.columns(2)
        payment = a4.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="f_pay",
        )
        monthly = a5.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5, key="f_monthly")
        total_charges = tenure * monthly  # Auto-calculated

        divider()
        st.markdown("**Phone Services**")
        p1, p2 = st.columns(2)
        phone = p1.selectbox("Phone Service", ["Yes", "No"], key="f_phone")
        multi = p2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="f_multi")

        divider()
        st.markdown("**Internet Services**")
        i1, i2, i3 = st.columns(3)
        internet = i1.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="f_internet")
        online_sec = i2.selectbox("Online Security", ["Yes", "No", "No internet service"], key="f_osec")
        online_bck = i3.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="f_obck")

        i4, i5, i6 = st.columns(3)
        device = i4.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="f_dev")
        tech = i5.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="f_tech")
        streaming_tv = i6.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="f_stv")

        _, i8, _ = st.columns([1, 1, 1])
        streaming_mv = i8.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="f_smv")

        divider()
        st.caption(f"💡 **Auto-calculated Total Charges:** ${total_charges:,.2f} (Tenure × Monthly)")

        submitted = st.form_submit_button(
            "🎯 Predict Churn Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        # Build input DataFrame
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multi,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bck,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mv,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
        }
        input_df = pd.DataFrame([input_data])

        with st.spinner("Running prediction…"):
            try:
                pred, proba = utils.predict_churn(bundle, input_df)
            except Exception as e:
                st.error(f"Prediction error: {e}", icon="❌")
                return

        divider()
        st.markdown("### 🎯 Prediction Result")

        res_col, gauge_col = st.columns([1, 1], gap="large")

        with res_col:
            if pred == 1:
                st.markdown(
                    f"""
                    <div class="result-high">
                        <div class="result-title">🔴 High Churn Risk</div>
                        <div class="result-prob">
                            This customer is likely to churn.<br/>
                            <strong>Confidence: {proba*100:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-low">
                        <div class="result-title">🟢 Likely to Stay</div>
                        <div class="result-prob">
                            This customer is unlikely to churn.<br/>
                            <strong>Confidence: {(1 - proba)*100:.1f}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if proba is not None:
                st.markdown("")
                st.progress(proba, text=f"Churn probability: {proba*100:.1f}%")

            st.markdown("")
            with st.expander("📋 Input Summary"):
                summary_df = pd.DataFrame(list(input_data.items()), columns=["Feature", "Value"])
                summary_df["Value"] = summary_df["Value"].astype(str)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with gauge_col:
            if proba is not None:
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=proba * 100,
                        number={"suffix": "%", "font": {"size": 36, "color": "#e2e8f0"}},
                        delta={
                            "reference": 50,
                            "increasing": {"color": COLOR_CHURN},
                            "decreasing": {"color": COLOR_STAY},
                        },
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                            "bar": {"color": COLOR_CHURN if pred == 1 else COLOR_STAY},
                            "bgcolor": "#0f172a",
                            "borderwidth": 2,
                            "bordercolor": "#334155",
                            "steps": [
                                {"range": [0, 30], "color": "#052e16"},
                                {"range": [30, 60], "color": "#422006"},
                                {"range": [60, 100], "color": "#450a0a"},
                            ],
                            "threshold": {
                                "line": {"color": "#f59e0b", "width": 3},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                        title={
                            "text": "Churn Probability",
                            "font": {"size": 16, "color": "#94a3b8"},
                        },
                    )
                )
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#e2e8f0"),
                    height=380,
                    margin=dict(l=30, r=30, t=60, b=30),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Retention recommendations
            with st.expander("💡 Retention Recommendations", expanded=True):
                if pred == 1:
                    st.markdown(
                        """
                        **Suggested retention actions for this customer:**
                        - 🔒 Offer a **discounted long-term contract** (1 or 2 year)
                        - 💳 Migrate from **electronic check** to auto-payment
                        - 🎁 Provide a **loyalty discount** on monthly charges
                        - 📞 Schedule a **proactive support call**
                        - 🌐 Review and upgrade **internet service quality**
                        """
                    )
                else:
                    st.markdown(
                        """
                        **This customer appears satisfied. To keep them engaged:**
                        - 🏅 Enroll in a **loyalty rewards program**
                        - 📦 Upsell **premium add-ons** (streaming, security)
                        - 📣 Share **referral incentives**
                        """
                    )


# =============================================================================
# ROUTER
# =============================================================================

page_map = {
    "🏠  Home": page_home,
    "📂  Dataset Explorer": page_dataset_explorer,
    "📈  EDA Dashboard": page_eda,
    "🤖  Model Training": page_model_training,
    "🏆  Model Comparison": page_model_comparison,
    "🎯  Prediction System": page_prediction,
}

page_map[page]()
