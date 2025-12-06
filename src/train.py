"""
VitalSense Model Training Script (Weighted & Robust)

This script loads the unified dataset, applies medical-aware preprocessing,
applies sample weighting for high-quality rows, scales features using
pre-computed statistics, trains XGBoost models for Diabetes and Heart Disease,
evaluates them, and saves the models.

"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_PROCESSED_PATH = r"D:\brototype\BW1\VitalSense\data\processed"
MODEL_PATH = r"D:\brototype\BW1\VitalSense\model"

RAW_FILE = os.path.join(DATA_PROCESSED_PATH, "unified_health_dataset.csv")
STATS_FILE = os.path.join(DATA_PROCESSED_PATH, "scaler_stats.json")


# Data Loading & Preprocessing
def load_and_prep_data():
    """
    Load dataset, clean numeric fields, assign sample weights, and apply
    medical-aware filling rules. This must match the logic used during
    live monitoring.

    Returns:
        pd.DataFrame: The cleaned and weighted dataset.
    """
    logger.info(f"Loading raw data from: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)

    numeric_cols = ["Age", "BloodPressure", "Cholesterol",
                    "Glucose", "BMI", "HbA1c"]

    # Clean numeric columns
    logger.info("Cleaning numeric columns...")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    # Standard Training (No Sample Weights due to bias in valid subset)
    logger.info(f"Total rows: {len(df)}")
    
    # Encode Gender
    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"]
            .astype(str)
            .str.lower()
            .map({"male": 1, "m": 1, "female": 0, "f": 0})
            .fillna(0)
            .astype(int)
        )

    # Medical-aware rules
    logger.info("Applying medical-aware filling rules...")

    df["Age"].fillna(df["Age"].median(), inplace=True)

    df.loc[(df["BloodPressure"] < 60) | (df["BloodPressure"] > 200), "BloodPressure"] = np.nan
    df["BloodPressure"].fillna(df["BloodPressure"].median(), inplace=True)

    df.loc[(df["Glucose"] < 40) | (df["Glucose"] > 500), "Glucose"] = np.nan
    df["Glucose"].fillna(df["Glucose"].median(), inplace=True)

    df["Cholesterol"].fillna(df["Cholesterol"].median(), inplace=True)

    df["BMI"] = df.groupby(
        pd.cut(df["Age"], bins=[0, 20, 40, 60, 80, 120])
    )["BMI"].transform(lambda x: x.fillna(x.mean()))
    df["BMI"].fillna(df["BMI"].mean(), inplace=True)

    df["HbA1c"].fillna(5.7, inplace=True)

    return df


# Scaling
def scale_data(df):
    """
    Scale selected columns using precomputed mean/std stored in JSON.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Scaled dataset.
    """
    logger.info(f"Loading scaler stats from: {STATS_FILE}")

    with open(STATS_FILE, "r") as f:
        stats = json.load(f)

    scaled_cols = ["BloodPressure", "Cholesterol", "Glucose", "BMI", "HbA1c"]

    logger.info("Scaling features...")
    for col in scaled_cols:
        mean = stats[col]["mean"]
        std = stats[col]["std"] or 1.0
        df[col] = (df[col] - mean) / std

    return df


# Model Training
def train_xgb_model(X_train, y_train):
    """
    Train a standard XGBoost model with automatic imbalance handling.

    Args:
        X_train (pd.DataFrame)
        y_train (pd.Series)

    Returns:
        XGBClassifier: Trained model.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    logger.info(f"Training XGBoost (scale_pos_weight={scale_pos:.2f})...")

    model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )

    model.fit(X_train, y_train)
    return model


# Evaluation
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and log metrics.

    Args:
        model (XGBClassifier)
        X_test (pd.DataFrame)
        y_test (pd.Series)
        model_name (str)

    Returns:
        tuple: (accuracy, f1, auc)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f"\n===== {model_name} Metrics =====")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")
    logger.info(f"ROC-AUC  : {auc:.4f}")

    return acc, f1, auc


# Saving
def save_model(model, filename):
    """
    Save a trained model to disk.

    Args:
        model (XGBClassifier)
        filename (str)
    """
    os.makedirs(MODEL_PATH, exist_ok=True)
    path = os.path.join(MODEL_PATH, filename)

    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")


# Main Workflow
def main():
    """Main execution entry point for training both models."""
    logger.info("Starting weighted model training pipeline...")

    df = load_and_prep_data()
    df = scale_data(df)

    features = ["Age", "Gender", "BloodPressure",
                "Cholesterol", "Glucose", "BMI", "HbA1c"]


    # Diabetes Model
    logger.info("\nTraining Diabetes Model...")
    y_dia = df["Diabetes"].fillna(0).astype(int)
    X_dia = df[features]

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_dia, y_dia, test_size=0.2, random_state=42, stratify=y_dia
    )

    dia_model = train_xgb_model(X_train_d, y_train_d)
    evaluate_model(dia_model, X_test_d, y_test_d, "Diabetes Model")
    save_model(dia_model, "diabetes_xgb.pkl")


    # Heart Disease Model
    logger.info("\nTraining Heart Disease Model...")
    y_heart = df["HeartDisease"].fillna(0).astype(int)
    X_heart = df[features]

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
    )

    heart_model = train_xgb_model(X_train_h, y_train_h)
    evaluate_model(heart_model, X_test_h, y_test_h, "Heart Disease Model")
    save_model(heart_model, "heart_xgb.pkl")

    logger.info("\nAll weighted models trained and saved successfully!")


if __name__ == "__main__":
    main()