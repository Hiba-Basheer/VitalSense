"""
Data Unification Script for VitalSense

This script loads raw diabetes and heart disease datasets, standardizes
column names and formats, merges them into a unified dataset, and saves
the processed output for model training.

"""

import os
import logging
import pandas as pd

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# File Paths
DATA_RAW_PATH = r"D:\brototype\BW1\VitalSense\data\raw"
DATA_PROCESSED_PATH = r"D:\brototype\BW1\VitalSense\data\processed"


def load_datasets():
    """
    Load raw datasets from disk.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame):
        The diabetes1, diabetes2, and heart datasets.
    """
    logger.info("Loading datasets...")

    df_d1 = pd.read_csv(os.path.join(DATA_RAW_PATH, "diabetes.csv"))
    df_d2 = pd.read_csv(os.path.join(DATA_RAW_PATH, "diabetes_dataset.csv"))
    df_h = pd.read_csv(os.path.join(DATA_RAW_PATH, "heart.csv"))

    logger.info("Datasets loaded successfully.")
    return df_d1, df_d2, df_h


def standardize_diabetes_d1(df):
    """
    Standardize the first diabetes dataset (diabetes.csv).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Standardized dataframe.
    """
    logger.info("Standardizing diabetes.csv...")

    df = df.rename(columns={
        "stab.glu": "Glucose",
        "bp.1s": "BloodPressure",
        "chol": "Cholesterol",
        "age": "Age",
        "gender": "Gender"
    })

    df["Diabetes"] = 1
    df["HeartDisease"] = None
    df["BMI"] = None
    df["HbA1c"] = None

    return df[[
        "Age", "Gender", "BloodPressure", "Cholesterol",
        "Glucose", "BMI", "HbA1c", "Diabetes", "HeartDisease"
    ]]


def standardize_diabetes_d2(df):
    """
    Standardize the second diabetes dataset (diabetes_dataset.csv).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Standardized dataframe.
    """
    logger.info("Standardizing diabetes_dataset.csv...")

    df = df.rename(columns={
        "blood_glucose_level": "Glucose",
        "hbA1c_level": "HbA1c",
        "bmi": "BMI",
        "age": "Age",
        "gender": "Gender",
        "heart_disease": "HeartDisease",
        "diabetes": "Diabetes"
    })

    df["BloodPressure"] = None
    df["Cholesterol"] = None

    return df[[
        "Age", "Gender", "BloodPressure", "Cholesterol",
        "Glucose", "BMI", "HbA1c", "Diabetes", "HeartDisease"
    ]]


def standardize_heart(df):
    """
    Standardize the heart dataset (heart.csv).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Standardized dataframe.
    """
    logger.info("Standardizing heart.csv...")

    df = df.rename(columns={
        "Sex": "Gender",
        "RestingBP": "BloodPressure",
        "Cholesterol": "Cholesterol",
        "HeartDisease": "HeartDisease"
    })

    df["Glucose"] = None
    df["BMI"] = None
    df["HbA1c"] = None
    df["Diabetes"] = None

    return df[[
        "Age", "Gender", "BloodPressure", "Cholesterol",
        "Glucose", "BMI", "HbA1c", "Diabetes", "HeartDisease"
    ]]


def merge_and_save(df1, df2, dfh):
    """
    Merge datasets and save to processed directory.

    Args:
        df1 (pd.DataFrame)
        df2 (pd.DataFrame)
        dfh (pd.DataFrame)
    """
    logger.info("Merging datasets...")

    df_all = pd.concat([df1, df2, dfh], ignore_index=True)
    logger.info(f"Merged dataset shape: {df_all.shape}")

    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED_PATH, "unified_health_dataset.csv")

    df_all.to_csv(output_path, index=False)
    logger.info(f"Unified dataset saved to: {output_path}")


def main():
    """Main execution function."""
    df_d1, df_d2, df_h = load_datasets()

    df_d1_std = standardize_diabetes_d1(df_d1)
    df_d2_std = standardize_diabetes_d2(df_d2)
    df_h_std = standardize_heart(df_h)

    merge_and_save(df_d1_std, df_d2_std, df_h_std)


if __name__ == "__main__":
    main()
