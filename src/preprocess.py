"""
preprocess.py
Data loading, cleaning, leakage removal, feature engineering, and temporal splitting.
"""

import os
import glob
from typing import Tuple, List, Dict, Set
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Default feature configuration based on R2 leakage avoidance
DEFAULT_CAT_FEATURES = [
    "agency",
    "province_en",
    "vehicle_type",
    "weather_condition",
    "road_description",
    "slope_description",
]

DEFAULT_NUM_FEATURES = [
    "number_of_vehicles_involved",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_night",
    "is_songkran",
    "is_newyear",
]

TARGET = "fatal"


def load_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Loads dataset from provided path, local data/raw directory, or downloads via kagglehub.
    """
    if file_path and os.path.exists(file_path):
        print(f"[Data] Loading dataset from: {file_path}")
        return pd.read_csv(file_path)

    # Check local data/raw folder
    raw_files = glob.glob(os.path.join("data", "raw", "*.csv"))
    if raw_files:
        print(f"[Data] Found local raw dataset: {raw_files[0]}")
        return pd.read_csv(raw_files[0])

    # Download from Kaggle via kagglehub if available
    try:
        import kagglehub
        print("[Data] Downloading dataset from Kaggle (thaweewatboy/thailand-road-accident-2019-2022)...")
        path = kagglehub.dataset_download("thaweewatboy/thailand-road-accident-2019-2022")
        downloaded_csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if downloaded_csvs:
            print(f"[Data] Loaded from Kaggle cache: {downloaded_csvs[0]}")
            return pd.read_csv(downloaded_csvs[0])
    except Exception as e:
        print(f"[Warning] Could not download from Kagglehub: {e}")

    raise FileNotFoundError(
        "No CSV dataset found! Please place the dataset CSV in data/raw/ or install kagglehub."
    )


def clean_and_engineer_features(df: pd.DataFrame, use_accident_type: bool = False) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Cleans dataset, removes leakage columns, and creates temporal features.
    """
    # Filter valid incident datetimes
    df = df.copy()
    df["incident_datetime"] = pd.to_datetime(df["incident_datetime"], errors="coerce")
    df = df.dropna(subset=["incident_datetime"]).copy()
    
    t = df["incident_datetime"]
    df["year"] = t.dt.year
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 18) | (df["hour"] < 6)).astype(int)

    mmdd = t.dt.strftime("%m-%d")
    # Songkran 7 dangerous days (Apr 11 - 17)
    df["is_songkran"] = mmdd.between("04-11", "04-17").astype(int)
    # New Year 7 dangerous days (Dec 29 - Jan 4)
    df["is_newyear"] = ((mmdd >= "12-29") | (mmdd <= "01-04")).astype(int)

    # Target variable: binary fatality
    df[TARGET] = (df["number_of_fatalities"] > 0).astype(int)

    cat_features = list(DEFAULT_CAT_FEATURES)
    if use_accident_type and "accident_type" in df.columns:
        cat_features.append("accident_type")

    num_features = list(DEFAULT_NUM_FEATURES)

    return df, cat_features, num_features


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data chronologically:
    - Train: 2019-2021
    - Test:  2022
    """
    train = df[df["year"] <= 2021].copy()
    test = df[df["year"] == 2022].copy()
    return train, test


def group_rare_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_features: List[str],
    min_count: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """
    Groups categories that occur less than min_count in the training set into 'other'.
    Rules are derived strictly from train to prevent data leakage.
    """
    keep_levels = {}
    for col in cat_features:
        vc = train[col].fillna("unknown").value_counts()
        keep_levels[col] = set(vc[vc >= min_count].index)

    def _replace(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in cat_features:
            s = out[col].fillna("unknown")
            out[col] = s.where(s.isin(keep_levels[col]), "other")
        return out

    train_proc = _replace(train)
    test_proc = _replace(test)
    return train_proc, test_proc, keep_levels


def build_pipeline_preprocessor(cat_features: List[str], num_features: List[str]) -> ColumnTransformer:
    """
    Builds a ColumnTransformer for one-hot encoding categorical features
    and standard-scaling numeric features.
    """
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features),
    ])
