"""Core poisoning logic from Section 5.1 / 5.7 of the paper.

Equations implemented
---------------------
Feature poisoning  (Eq. 4):
    X_poisoned[i, j] = X[i, j] + ε,   ε ~ N(0, σ)

Label flipping     (Eq. 5):
    y_poisoned[i] = random(y)          (uniform over existing classes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def poison_data(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    poison_fraction: float = 0.10,
    noise_std: float = 0.5,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply data poisoning to a fraction of the dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Label vector.
    poison_fraction : float, default 0.10
        Fraction of samples to poison (paper uses 10 %).
    noise_std : float, default 0.5
        Standard deviation σ of the Gaussian noise (paper: σ = 0.5).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X_out : ndarray
        Feature matrix with poisoned rows.
    y_out : ndarray
        Label vector with flipped labels for poisoned rows.
    poison_mask : ndarray of bool
        Boolean mask indicating which rows were poisoned.
    """
    rng = np.random.default_rng(random_state)

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y).copy()

    n_samples = X_arr.shape[0]
    n_poison = int(n_samples * poison_fraction)

    # Randomly select indices to poison
    poison_idx = rng.choice(n_samples, size=n_poison, replace=False)
    poison_mask = np.zeros(n_samples, dtype=bool)
    poison_mask[poison_idx] = True

    # Eq. 4 – add Gaussian noise to features
    noise = rng.normal(loc=0.0, scale=noise_std, size=X_arr[poison_idx].shape)
    X_arr[poison_idx] += noise

    # Eq. 5 – flip labels to a random (different) class
    unique_classes = np.unique(y_arr)
    for idx in poison_idx:
        choices = unique_classes[unique_classes != y_arr[idx]]
        y_arr[idx] = rng.choice(choices)

    return X_arr, y_arr, poison_mask
