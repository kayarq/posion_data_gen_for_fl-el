"""Preprocessing pipeline from Section 4 of the paper.

Steps
-----
1. Load CSV(s) and concatenate.
2. Handle missing values (drop rows with NaN).
3. Encode target labels with LabelEncoder.
4. (Optional) Balance classes with ADASYN  (Eq. 1).
5. Select top-k features via Random Forest importance (Eqs. 2–3).
6. Train/test split (80/20).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _load_csvs(paths: list[str | Path], sample_n: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """Load one or more CSV files and optionally sample *n* rows."""
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state)
    return df


def preprocess(
    csv_paths: list[str | Path],
    target_col: str = "label",
    top_k_features: int = 15,
    test_size: float = 0.20,
    balance: bool = True,
    sample_n: int | None = None,
    keep_classes: list[str] | None = None,
    random_state: int = 42,
) -> dict:
    """Full preprocessing pipeline.

    Parameters
    ----------
    csv_paths : list of paths
        One or more CSV files (e.g. the 60+ CICIoT-2023 parts).
    target_col : str
        Name of the target / label column.
    top_k_features : int, default 15
        Number of features to keep (paper uses 15).
    test_size : float, default 0.20
        Fraction reserved for testing (paper: 20 %).
    balance : bool, default True
        Whether to apply ADASYN to balance minority classes.
    sample_n : int or None
        If set, subsample the combined data to this many rows before
        processing (paper used 1 000 000).
    keep_classes : list of str or None
        If given, filter the dataset to only these target classes
        (paper kept 5 important classes).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        feature_names, label_encoder, selected_features
    """
    # 1. Load
    df = _load_csvs(csv_paths, sample_n=sample_n, random_state=random_state)

    # 2. Handle missing / infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Optional class filter
    if keep_classes is not None:
        df = df[df[target_col].isin(keep_classes)]

    # 3. Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    X = df.drop(columns=[target_col])

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # 5. Feature selection via RF importance (Eqs. 2–3)
    if top_k_features and top_k_features < X.shape[1]:
        rf_selector = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1,
        )
        rf_selector.fit(X, y)
        importances = rf_selector.feature_importances_
        top_idx = np.argsort(importances)[::-1][:top_k_features]
        selected_features = list(X.columns[top_idx])
        X = X[selected_features]
    else:
        selected_features = list(X.columns)

    # 6. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # 4. Balance with ADASYN (Eq. 1) – applied only to train set
    if balance:
        try:
            from imblearn.over_sampling import ADASYN

            ada = ADASYN(random_state=random_state)
            X_train, y_train = ada.fit_resample(X_train, y_train)
        except ValueError:
            # ADASYN can fail when some classes have too few samples; skip silently
            pass

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": selected_features,
        "label_encoder": le,
    }
