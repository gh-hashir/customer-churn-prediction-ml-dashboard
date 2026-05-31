# =============================================================================
# utils.py — Shared utility functions for the Churn Prediction Dashboard
# =============================================================================
# Responsibilities:
#   - Data loading & caching
#   - Preprocessing pipeline construction
#   - Model training (with SMOTE, no leakage)
#   - Metrics computation
#   - Model serialization / deserialization
# =============================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_churn_model.pkl")

# ── Column definitions ─────────────────────────────────────────────────────────
NUMERICAL_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]

# ── Available models ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Logistic Regression": LogisticRegression(random_state=42, solver="liblinear", max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "XGBoost": None,  # Lazy-loaded to avoid hard import error if xgboost not installed
}


def _get_xgboost():
    """Lazy-load XGBClassifier to avoid import errors if not installed."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=100,
        )
    except ImportError:
        return None


def get_model_instance(model_name: str):
    """Return a fresh (unfitted) estimator instance for the given name."""
    if model_name == "XGBoost":
        clf = _get_xgboost()
        if clf is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return clf
    instance = MODEL_REGISTRY.get(model_name)
    if instance is None:
        raise ValueError(f"Unknown model: {model_name}")
    # Return a fresh clone so session state doesn't share references
    from sklearn.base import clone
    return clone(instance)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load and lightly clean the Telco Customer Churn CSV.
    - Converts TotalCharges to numeric (coerce errors → NaN → 0)
    - Maps Churn: Yes→1, No→0
    Returns the cleaned DataFrame.
    """
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found at `{DATA_PATH}`. "
            "Please ensure the CSV is placed in the `data/` folder."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # Fix TotalCharges (stored as object with spaces for new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    return df


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer that:
    - StandardScales numerical columns
    - OneHotEncodes categorical columns
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_after_encoding(preprocessor: ColumnTransformer) -> list:
    """Extract human-readable feature names after fitting the ColumnTransformer."""
    num_features = NUMERICAL_COLS.copy()
    cat_features = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
    )
    return num_features + cat_features


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(model_name: str, df: pd.DataFrame):
    """
    Full training pipeline:
    1. Stratified 80/20 train-test split
    2. Preprocessor fit on train only
    3. SMOTE applied to processed training data only
    4. Model trained on resampled data
    5. Metrics computed on held-out test set

    Returns:
        results (dict): metrics + confusion matrix + roc/pr curve data
        pipeline (fitted Pipeline): for serialization / prediction
    """
    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    y = df[TARGET_COL]

    # ── Stratified split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Preprocess ──
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # ── SMOTE on train only ──
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    # ── Train model ──
    clf = get_model_instance(model_name)
    clf.fit(X_train_res, y_train_res)

    # ── Metrics ──
    results = compute_metrics(clf, X_test_proc, y_test, model_name)

    # ── Save preprocessor + classifier together for prediction ──
    # We wrap in a simple dict (not imblearn Pipeline) since SMOTE is train-only
    bundle = {"preprocessor": preprocessor, "classifier": clf}

    return results, bundle


def compute_metrics(clf, X_test_proc, y_test, model_name: str) -> dict:
    """Compute classification metrics and curve data for one trained classifier."""
    y_pred = clf.predict(X_test_proc)
    y_proba = (
        clf.predict_proba(X_test_proc)[:, 1]
        if hasattr(clf, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC curve
    roc_data = None
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # Precision-Recall curve
    pr_data = None
    if y_proba is not None:
        precision_c, recall_c, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        pr_data = {
            "precision": precision_c.tolist(),
            "recall": recall_c.tolist(),
            "ap": ap,
        }

    return {
        "model_name": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "confusion_matrix": cm,
        "roc_data": roc_data,
        "pr_data": pr_data,
    }


# =============================================================================
# MODEL SERIALIZATION
# =============================================================================

def save_model(bundle: dict, path: str = MODEL_PATH):
    """Serialize the preprocessor+classifier bundle to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


@st.cache_resource(show_spinner=False)
def load_model(path: str = MODEL_PATH):
    """Load the serialized model bundle from disk (cached across reruns)."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# =============================================================================
# PREDICTION HELPER
# =============================================================================

def predict_churn(bundle: dict, input_df: pd.DataFrame):
    """
    Run prediction using the stored preprocessor + classifier.
    input_df: single-row DataFrame with raw (unencoded) feature values.
    Returns (prediction: int, probability: float).
    """
    preprocessor = bundle["preprocessor"]
    clf = bundle["classifier"]

    X_proc = preprocessor.transform(input_df)
    pred = clf.predict(X_proc)[0]
    proba = (
        clf.predict_proba(X_proc)[0][1]
        if hasattr(clf, "predict_proba")
        else None
    )
    return int(pred), float(proba) if proba is not None else None


# =============================================================================
# FEATURE IMPORTANCE HELPER
# =============================================================================

def get_feature_importance(preprocessor: ColumnTransformer, clf) -> pd.Series:
    """
    Extract feature importances from a tree-based model (RF, GB, XGB).
    Falls back to coefficient magnitudes for linear models.
    Returns a sorted pd.Series (top features first).
    """
    feature_names = get_feature_names_after_encoding(preprocessor)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return pd.Series(dtype=float)

    return (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
    )
