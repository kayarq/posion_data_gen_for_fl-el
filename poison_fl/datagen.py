"""Synthetic CICIoT-2023-like IoT traffic data generator.

Generates realistic network-traffic features with attack classes matching the
CICIoT-2023 dataset used in Al Dalaien et al. (Array 28, 2025).

Attack taxonomy (7 categories from CICIoT-2023):
    Benign, DDoS, DoS, Recon, Spoofing, Mirai, BruteForce, Web

Each class has a distinct statistical profile for the 46 network features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Feature names modelled after CICIoT-2023 ────────────────────────────────
FEATURES = [
    "flow_duration", "Header_Length", "Protocol_Type", "Duration", "Rate",
    "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "urg_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot_sum", "Min", "Max", "AVG", "Std",
    "Tot_size", "IAT", "Number", "Magnitude", "Radius",
    "Covariance", "Variance", "Weight",
]

# ── Per-class statistical profiles (mean, std for each feature) ──────────────
# These are hand-tuned to produce distributions that resemble real IoT traffic
# patterns. Each row: (mean, std) arrays of shape (n_features,).

def _class_profiles(n_features: int, rng: np.random.Generator) -> dict[str, dict]:
    """Return mean/std profiles for each traffic class."""
    return {
        "Benign": {
            "mean": np.array([
                0.5, 40, 6, 0.5, 50, 25, 25, 0.1, 0.2, 0.05,
                0.3, 0.8, 0.01, 0.01, 0.01, 15, 3, 1, 0.5,
                0.3, 0.4, 0.2, 0.01, 0.01, 0.02, 0.0, 0.7, 0.3,
                0.01, 0.02, 0.05, 1.0, 0.5,
                500, 40, 1500, 200, 150,
                800, 0.05, 10, 50, 5,
                10, 20, 1.0,
            ][:n_features]),
            "std": np.array([
                0.3, 15, 2, 0.3, 30, 15, 15, 0.1, 0.15, 0.05,
                0.2, 0.15, 0.01, 0.01, 0.01, 8, 2, 1, 0.5,
                0.2, 0.2, 0.15, 0.01, 0.01, 0.02, 0.01, 0.15, 0.15,
                0.01, 0.02, 0.05, 0.1, 0.2,
                300, 20, 500, 100, 100,
                400, 0.03, 5, 30, 3,
                8, 15, 0.5,
            ][:n_features]),
        },
        "DDoS": {
            "mean": np.array([
                0.01, 60, 6, 0.01, 5000, 4500, 500, 0.05, 0.9, 0.1,
                0.1, 0.95, 0.0, 0.0, 0.0, 800, 500, 2, 5,
                0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2,
                0.0, 0.0, 0.0, 1.0, 0.5,
                50000, 40, 60, 45, 5,
                60000, 0.001, 1000, 500, 2,
                1, 3, 10.0,
            ][:n_features]),
            "std": np.array([
                0.005, 10, 1, 0.005, 2000, 1500, 300, 0.05, 0.1, 0.1,
                0.1, 0.05, 0.01, 0.01, 0.01, 200, 150, 2, 3,
                0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1,
                0.01, 0.01, 0.01, 0.1, 0.2,
                20000, 10, 15, 10, 3,
                25000, 0.001, 300, 200, 1,
                1, 2, 3.0,
            ][:n_features]),
        },
        "DoS": {
            "mean": np.array([
                0.05, 55, 6, 0.05, 2000, 1800, 200, 0.3, 0.7, 0.2,
                0.2, 0.8, 0.0, 0.0, 0.0, 400, 200, 5, 10,
                0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.15,
                0.0, 0.0, 0.01, 1.0, 0.5,
                20000, 50, 80, 60, 10,
                25000, 0.005, 500, 300, 3,
                3, 8, 7.0,
            ][:n_features]),
            "std": np.array([
                0.03, 12, 1.5, 0.03, 800, 700, 150, 0.15, 0.15, 0.15,
                0.15, 0.1, 0.01, 0.01, 0.01, 150, 100, 3, 5,
                0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1,
                0.01, 0.01, 0.01, 0.1, 0.2,
                10000, 15, 20, 15, 5,
                12000, 0.003, 150, 100, 2,
                2, 5, 2.0,
            ][:n_features]),
        },
        "Recon": {
            "mean": np.array([
                0.8, 44, 6, 0.8, 100, 80, 20, 0.1, 0.8, 0.6,
                0.05, 0.3, 0.0, 0.0, 0.0, 20, 15, 0.5, 8,
                0.05, 0.02, 0.3, 0.0, 0.0, 0.1, 0.0, 0.6, 0.3,
                0.0, 0.0, 0.1, 1.0, 0.5,
                200, 40, 60, 48, 8,
                300, 0.1, 50, 20, 8,
                15, 25, 2.0,
            ][:n_features]),
            "std": np.array([
                0.4, 10, 2, 0.4, 60, 50, 15, 0.1, 0.15, 0.2,
                0.05, 0.2, 0.01, 0.01, 0.01, 10, 8, 0.5, 4,
                0.05, 0.02, 0.15, 0.01, 0.01, 0.05, 0.01, 0.15, 0.15,
                0.01, 0.01, 0.05, 0.1, 0.2,
                100, 10, 15, 12, 5,
                150, 0.05, 20, 10, 4,
                10, 15, 1.0,
            ][:n_features]),
        },
        "Spoofing": {
            "mean": np.array([
                0.3, 50, 17, 0.3, 300, 200, 100, 0.2, 0.5, 0.1,
                0.1, 0.6, 0.0, 0.0, 0.0, 50, 30, 3, 2,
                0.1, 0.05, 0.5, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3,
                0.1, 0.2, 0.1, 1.0, 0.5,
                1500, 30, 200, 80, 60,
                2000, 0.02, 30, 80, 10,
                20, 40, 3.0,
            ][:n_features]),
            "std": np.array([
                0.2, 12, 3, 0.2, 150, 100, 60, 0.15, 0.2, 0.1,
                0.1, 0.2, 0.01, 0.01, 0.01, 25, 15, 2, 2,
                0.1, 0.05, 0.2, 0.01, 0.01, 0.01, 0.01, 0.15, 0.15,
                0.05, 0.1, 0.05, 0.1, 0.2,
                800, 15, 80, 40, 30,
                1000, 0.01, 15, 40, 5,
                12, 25, 1.5,
            ][:n_features]),
        },
        "Mirai": {
            "mean": np.array([
                0.02, 48, 6, 0.02, 3000, 2500, 500, 0.1, 0.85, 0.05,
                0.1, 0.9, 0.0, 0.0, 0.0, 600, 400, 1, 3,
                0.0, 0.0, 0.0, 0.3, 0.0, 0.2, 0.1, 0.7, 0.2,
                0.0, 0.0, 0.0, 1.0, 0.5,
                35000, 44, 52, 47, 3,
                40000, 0.002, 700, 400, 2,
                2, 4, 8.0,
            ][:n_features]),
            "std": np.array([
                0.01, 8, 1, 0.01, 1200, 1000, 300, 0.1, 0.1, 0.05,
                0.1, 0.05, 0.01, 0.01, 0.01, 200, 120, 1, 2,
                0.01, 0.01, 0.01, 0.15, 0.01, 0.1, 0.05, 0.1, 0.1,
                0.01, 0.01, 0.01, 0.1, 0.2,
                15000, 8, 10, 6, 2,
                18000, 0.001, 200, 150, 1,
                1, 3, 2.5,
            ][:n_features]),
        },
        "BruteForce": {
            "mean": np.array([
                2.0, 45, 6, 2.0, 20, 15, 5, 0.05, 0.6, 0.3,
                0.2, 0.5, 0.0, 0.0, 0.0, 30, 20, 1, 5,
                0.1, 0.1, 0.0, 0.1, 0.0, 0.5, 0.0, 0.8, 0.1,
                0.0, 0.0, 0.0, 1.0, 0.5,
                400, 42, 100, 55, 20,
                600, 0.5, 20, 30, 6,
                8, 12, 1.5,
            ][:n_features]),
            "std": np.array([
                1.0, 10, 1.5, 1.0, 12, 10, 4, 0.05, 0.2, 0.15,
                0.15, 0.2, 0.01, 0.01, 0.01, 15, 10, 1, 3,
                0.1, 0.1, 0.01, 0.05, 0.01, 0.2, 0.01, 0.1, 0.05,
                0.01, 0.01, 0.01, 0.1, 0.2,
                200, 10, 30, 15, 10,
                300, 0.2, 8, 15, 3,
                5, 8, 0.8,
            ][:n_features]),
        },
        "Web": {
            "mean": np.array([
                1.5, 120, 6, 1.5, 80, 50, 30, 0.2, 0.4, 0.1,
                0.6, 0.7, 0.0, 0.0, 0.0, 40, 10, 2, 1,
                0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1,
                0.0, 0.0, 0.0, 1.0, 0.5,
                3000, 60, 8000, 1500, 2000,
                5000, 0.08, 15, 100, 12,
                30, 50, 2.0,
            ][:n_features]),
            "std": np.array([
                0.8, 40, 1.5, 0.8, 40, 25, 15, 0.15, 0.2, 0.1,
                0.2, 0.15, 0.01, 0.01, 0.01, 20, 5, 2, 1,
                0.15, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05,
                0.01, 0.01, 0.01, 0.1, 0.2,
                1500, 20, 3000, 800, 1000,
                2500, 0.04, 8, 50, 6,
                18, 30, 1.0,
            ][:n_features]),
        },
    }


def generate_data(
    n_samples: int = 100_000,
    classes: list[str] | None = None,
    class_weights: dict[str, float] | None = None,
    imbalance_ratio: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic CICIoT-2023-like IoT traffic data.

    Parameters
    ----------
    n_samples : int, default 100_000
        Total number of samples to generate.
    classes : list of str or None
        Which classes to include.  Defaults to all 8 (Benign + 7 attacks).
        Paper kept 5 important classes.
    class_weights : dict or None
        Relative weight per class (will be normalised).  If *None* and
        *imbalance_ratio* is also *None*, classes are balanced.
    imbalance_ratio : float or None
        If set, Benign gets this fraction and attacks share the rest
        (e.g. 0.60 means 60 % benign).  Ignored when *class_weights* is given.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns = FEATURES + ["label"].
    """
    rng = np.random.default_rng(random_state)
    n_features = len(FEATURES)
    profiles = _class_profiles(n_features, rng)

    if classes is None:
        classes = list(profiles.keys())

    # Determine per-class sample counts
    if class_weights is not None:
        total_w = sum(class_weights[c] for c in classes)
        counts = {c: max(1, int(n_samples * class_weights[c] / total_w)) for c in classes}
    elif imbalance_ratio is not None:
        n_benign = int(n_samples * imbalance_ratio) if "Benign" in classes else 0
        attack_classes = [c for c in classes if c != "Benign"]
        n_per_attack = max(1, (n_samples - n_benign) // max(len(attack_classes), 1))
        counts = {c: n_per_attack for c in attack_classes}
        if "Benign" in classes:
            counts["Benign"] = n_benign
    else:
        n_per = max(1, n_samples // len(classes))
        counts = {c: n_per for c in classes}

    # Generate samples per class
    frames = []
    for cls in classes:
        n = counts[cls]
        p = profiles[cls]
        data = rng.normal(loc=p["mean"], scale=p["std"], size=(n, n_features))

        # Clamp binary/flag features to [0, 1]
        binary_cols = [i for i, f in enumerate(FEATURES)
                       if "flag" in f.lower() or f in (
                           "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
                           "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
                           "IPv", "LLC")]
        for col in binary_cols:
            if col < n_features:
                data[:, col] = np.clip(data[:, col], 0, 1)

        # Clamp count/size features to >= 0
        non_neg_cols = [i for i, f in enumerate(FEATURES)
                        if any(k in f.lower() for k in (
                            "count", "tot_", "min", "max", "avg", "std",
                            "size", "iat", "number", "magnitude", "radius",
                            "variance", "weight", "rate", "duration",
                            "header_length"))]
        for col in non_neg_cols:
            if col < n_features:
                data[:, col] = np.abs(data[:, col])

        df = pd.DataFrame(data, columns=FEATURES)
        df["label"] = cls
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    # Shuffle
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result
