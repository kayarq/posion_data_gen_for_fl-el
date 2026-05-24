# poison-fl

Poisoning data generation for Federated Learning, based on:

> Al Dalaien et al., *"A dual-aggregation approach to fortify federated learning
> against poisoning attacks in IoTs"*, Array **28** (2025) 100520.

## Install

```bash
pip install -e .
# with plotting extras:
pip install -e ".[plot]"
```

## What it does

Implements the paper's data-poisoning strategy (Section 5.1 / 5.7):

1. **Generate** (optional) – produce synthetic CICIoT-2023-like IoT traffic
   data from scratch (8 classes: Benign, DDoS, DoS, Recon, Spoofing, Mirai,
   BruteForce, Web) with realistic per-class feature distributions. No dataset
   download required.
2. **Preprocess** – load CSVs (or generated data), handle missing values,
   encode labels, balance minority classes with ADASYN (Eq. 1), select top-*k*
   features via Random Forest importance (Eqs. 2–3), then 80/20 train-test
   split.
3. **Distribute** – split training data equally among *N* FL clients (IID or
   non-IID).
4. **Poison** – for each client, randomly select a fraction (default 10 %) of
   samples and:
   - Add Gaussian noise to features: `X_poisoned[i,j] = X[i,j] + ε`, `ε ~ N(0, σ)` (Eq. 4)
   - Flip labels to a random different class (Eq. 5)

## CLI usage

**Generate synthetic data + poison (no dataset needed):**

```bash
poison-fl --generate \
    --generate-n 100000 \
    --n-clients 5 \
    --poison-fraction 0.10 \
    --output-dir poisoned_output/
```

**Use your own CSVs:**

```bash
poison-fl data/*.csv \
    --target-col label \
    --n-clients 5 \
    --poison-fraction 0.10 \
    --noise-std 0.5 \
    --top-k-features 15 \
    --output-dir poisoned_output/
```

This produces `poisoned_output/client_0.csv` … `client_4.csv` and `test.csv`.
Each client CSV includes an `is_poisoned` boolean column.

## Python API

**Generate + poison (no files needed):**

```python
from poison_fl import run_pipeline

result = run_pipeline(
    generate=True,
    generate_n=100_000,
    n_clients=5,
    poison_fraction=0.10,
    output_dir="poisoned_output",
)

for client in result["clients"]:
    print(f"Client {client['client_id']}: "
          f"{client['poison_mask'].sum()} poisoned / {len(client['y'])} total")
```

**From existing CSVs:**

```python
result = run_pipeline(
    csv_paths=["data/part1.csv", "data/part2.csv"],
    target_col="label",
    n_clients=5,
    poison_fraction=0.10,
    output_dir="poisoned_output",
)
```

### Generate data only

```python
from poison_fl import generate_data

df = generate_data(n_samples=50_000, classes=["Benign", "DDoS", "DoS", "Recon", "Spoofing"])
print(df["label"].value_counts())
```

### Use only the poisoning function

```python
from poison_fl import poison_data
import numpy as np

X = np.random.randn(1000, 15)
y = np.random.randint(0, 5, size=1000)

X_poisoned, y_poisoned, mask = poison_data(X, y, poison_fraction=0.10, noise_std=0.5)
```

## Parameters (paper defaults)

| Parameter | Default | Paper reference |
|---|---|---|
| `poison_fraction` | 0.10 | Section 5.1 – "10 % of the training dataset" |
| `noise_std` | 0.5 | Section 5.7.1 – "Gaussian noise (μ=0, σ=0.5)" |
| `top_k_features` | 15 | Section 4.2 – "15 features were selected" |
| `n_clients` | 5 | Section 5.2 – "five clients" |
| `test_size` | 0.20 | Section 4.1 – "80 %–20 % ratio" |
| `balance` | True | Section 4.1 – ADASYN (Eq. 1) |
