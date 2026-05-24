"""End-to-end pipeline that chains preprocessing → distribution → poisoning.

Matches Algorithm 1 (DAFL) steps 1–3 from Section 5.8 of the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poison_fl.poisoning import poison_data
from poison_fl.preprocessing import preprocess
from poison_fl.distribution import distribute_iid, distribute_non_iid


def run_pipeline(
    csv_paths: list[str | Path] | None = None,
    target_col: str = "label",
    n_clients: int = 5,
    poison_fraction: float = 0.10,
    noise_std: float = 0.5,
    top_k_features: int = 15,
    balance: bool = True,
    iid: bool = True,
    sample_n: int | None = None,
    keep_classes: list[str] | None = None,
    output_dir: str | Path | None = None,
    random_state: int = 42,
    generate: bool = False,
    generate_n: int = 100_000,
    generate_classes: list[str] | None = None,
    imbalance_ratio: float | None = None,
) -> dict:
    """Run the full poisoning-data generation pipeline.

    Parameters
    ----------
    csv_paths : list of paths or None
        Input CSV file(s).  Not needed when *generate* is True.
    target_col : str
        Name of the label column.
    n_clients : int
        Number of FL clients (paper: 5).
    poison_fraction : float
        Fraction of each client's data to poison (paper: 0.10).
    noise_std : float
        Gaussian noise σ for feature poisoning (paper: 0.5).
    top_k_features : int
        Number of features to keep via RF importance (paper: 15).
    balance : bool
        Apply ADASYN balancing.
    iid : bool
        True for IID distribution, False for non-IID.
    sample_n : int or None
        Subsample the raw data to this many rows.
    keep_classes : list of str or None
        Keep only these target classes.
    output_dir : path or None
        If given, save per-client CSVs here.
    random_state : int
        Global random seed.
    generate : bool, default False
        If True, generate synthetic CICIoT-2023-like data instead of
        reading CSV files.
    generate_n : int, default 100_000
        Number of samples to generate (only used when *generate* is True).
    generate_classes : list of str or None
        Classes to include in generated data.  Defaults to all 8.
    imbalance_ratio : float or None
        If set, Benign gets this fraction of generated data.

    Returns
    -------
    dict with keys:
        clients       – list of dicts (client_id, X, y, poison_mask)
        X_test        – test features
        y_test        – test labels
        feature_names – selected feature names
        label_encoder – fitted LabelEncoder
    """
    if generate:
        # Generate synthetic data, save to a temp CSV, then preprocess
        import tempfile
        from poison_fl.datagen import generate_data

        df = generate_data(
            n_samples=generate_n,
            classes=generate_classes,
            imbalance_ratio=imbalance_ratio,
            random_state=random_state,
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False)
        csv_paths = [tmp.name]
    elif not csv_paths:
        raise ValueError("Either provide csv_paths or set generate=True.")

    # Step 1 – preprocess
    prep = preprocess(
        csv_paths=csv_paths,
        target_col=target_col,
        top_k_features=top_k_features,
        balance=balance,
        sample_n=sample_n,
        keep_classes=keep_classes,
        random_state=random_state,
    )

    # Step 3 – distribute training data among clients
    dist_fn = distribute_iid if iid else distribute_non_iid
    clients_raw = dist_fn(
        prep["X_train"], prep["y_train"],
        n_clients=n_clients,
        random_state=random_state,
    )

    # Step 2 – poison each client's data
    clients = []
    for c in clients_raw:
        X_p, y_p, mask = poison_data(
            c["X"], c["y"],
            poison_fraction=poison_fraction,
            noise_std=noise_std,
            random_state=random_state + c["client_id"],
        )
        clients.append({
            "client_id": c["client_id"],
            "X": X_p,
            "y": y_p,
            "poison_mask": mask,
        })

    # Optionally save to disk
    if output_dir is not None:
        _save(clients, prep, output_dir)

    return {
        "clients": clients,
        "X_test": prep["X_test"],
        "y_test": prep["y_test"],
        "feature_names": prep["feature_names"],
        "label_encoder": prep["label_encoder"],
    }


def _save(clients: list[dict], prep: dict, output_dir: str | Path) -> None:
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    feature_names = prep["feature_names"]
    le = prep["label_encoder"]

    for c in clients:
        df = pd.DataFrame(c["X"], columns=feature_names)
        df["label"] = le.inverse_transform(c["y"])
        df["is_poisoned"] = c["poison_mask"]
        df.to_csv(out / f"client_{c['client_id']}.csv", index=False)

    df_test = pd.DataFrame(prep["X_test"], columns=feature_names)
    df_test["label"] = le.inverse_transform(prep["y_test"])
    df_test.to_csv(out / "test.csv", index=False)
