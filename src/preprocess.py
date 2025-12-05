"""
Health Dataset Preprocessing Script for VitalSense

This script cleans and preprocesses the unified health dataset.
It applies medical-aware rules, handles missing values, encodes
categorical fields, scales numeric values, and saves the processed file.

"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_PROCESSED_PATH = r"D:\brototype\BW1\VitalSense\data\processed"
INPUT_FILE = "unified_health_dataset.csv"
OUTPUT_FILE = "health_processed_dataset.csv"


def load_dataset():
    """
    Load the unified dataset from disk.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    path = os.path.join(DATA_PROCESSED_PATH, INPUT_FILE)
    logger.info(f"Loading dataset from: {path}")

    df = pd.read_csv(path)
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    return df


def encode_gender(df):
    """
    Convert gender strings to numeric values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with encoded Gender.
    """
    logger.info("Encoding gender column...")

    df["Gender"] = df["Gender"].replace({
        "M": 1, "Male": 1,
        "F": 0, "Female": 0
    })

    df["Gender"] = df["Gender"].fillna(0)
    return df


def clean_numeric_columns(df):
    """
    Convert numeric columns to valid ranges and handle invalid data.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info("Cleaning numeric columns...")

    numeric_cols = ["Age", "BloodPressure", "Cholesterol", "Glucose", "BMI", "HbA1c"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    return df


def apply_medical_rules(df):
    """
    Apply medical constraints and fill missing values based on domain logic.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Updated DataFrame.
    """
    logger.info("Applying medical-aware filling rules...")

    # Age: Median
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Blood Pressure: Valid range 60–200
    df.loc[(df["BloodPressure"] < 60) | (df["BloodPressure"] > 200), "BloodPressure"] = np.nan
    df["BloodPressure"].fillna(df["BloodPressure"].median(), inplace=True)

    # Glucose: Valid range 40–500
    df.loc[(df["Glucose"] < 40) | (df["Glucose"] > 500), "Glucose"] = np.nan
    df["Glucose"].fillna(df["Glucose"].median(), inplace=True)

    # Cholesterol: Fill with median
    df["Cholesterol"].fillna(df["Cholesterol"].median(), inplace=True)

    # BMI: Fill using mean BMI per age group
    df["BMI"] = df.groupby(
        pd.cut(df["Age"], bins=[0, 20, 40, 60, 80, 120])
    )["BMI"].transform(lambda x: x.fillna(x.mean()))

    # HbA1c: Fill with non-diabetic average
    df["HbA1c"].fillna(5.7, inplace=True)

    return df


def scale_features(df, stats):
    """
    Standardize selected numerical features using pre-calculated stats.

    Args:
        df (pd.DataFrame): Input DataFrame.
        stats (dict): Dictionary containing mean and std for each feature.

    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    logger.info("Scaling selected features...")

    scaled_cols = ["BloodPressure", "Cholesterol", "Glucose", "BMI", "HbA1c"]

    for col in scaled_cols:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        # Avoid division by zero
        if std == 0:
            std = 1.0
        
        df[col] = (df[col] - mean) / std

    return df



def save_dataset(df):
    """
    Save processed dataset to disk.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.
    """
    output_path = os.path.join(DATA_PROCESSED_PATH, OUTPUT_FILE)
    logger.info(f"Saving cleaned dataset to: {output_path}")

    df.to_csv(output_path, index=False)
    logger.info("Dataset saved successfully.")


def main():
    """
    Main execution workflow.
    """
    df = load_dataset()
    df = encode_gender(df)
    df = clean_numeric_columns(df)
    import json
    
    # Calculate stats on valid data (before imputation)
    stats = {}
    stats_cols = ["BloodPressure", "Cholesterol", "Glucose", "BMI", "HbA1c"]
    logger.info("Calculating stats on raw valid data...")
    for col in stats_cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }
    
    # Save stats
    stats_path = os.path.join(DATA_PROCESSED_PATH, "scaler_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Scaler stats saved to: {stats_path}")

    df = apply_medical_rules(df)
    df = scale_features(df, stats)

    save_dataset(df)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
