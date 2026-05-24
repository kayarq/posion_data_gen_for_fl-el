"""Command-line interface for poison-fl."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="poison-fl",
        description="Generate poisoned FL client datasets (Al Dalaien et al., 2025).",
    )
    parser.add_argument(
        "csv", nargs="*",
        help="One or more input CSV files (not needed with --generate).",
    )
    parser.add_argument(
        "-g", "--generate", action="store_true",
        help="Generate synthetic CICIoT-2023-like data instead of reading CSVs.",
    )
    parser.add_argument(
        "--generate-n", type=int, default=100_000,
        help="Number of synthetic samples to generate (default: 100000).",
    )
    parser.add_argument(
        "--generate-classes", nargs="+", default=None,
        help="Classes for synthetic data (default: all 8). "
             "Options: Benign DDoS DoS Recon Spoofing Mirai BruteForce Web",
    )
    parser.add_argument(
        "--imbalance-ratio", type=float, default=None,
        help="Fraction of Benign in generated data (e.g. 0.6 = 60%% benign).",
    )
    parser.add_argument(
        "-t", "--target-col", default="label",
        help="Name of the target/label column (default: label).",
    )
    parser.add_argument(
        "-n", "--n-clients", type=int, default=5,
        help="Number of FL clients (default: 5).",
    )
    parser.add_argument(
        "-p", "--poison-fraction", type=float, default=0.10,
        help="Fraction of data to poison per client (default: 0.10).",
    )
    parser.add_argument(
        "--noise-std", type=float, default=0.5,
        help="Gaussian noise std-dev σ (default: 0.5).",
    )
    parser.add_argument(
        "-k", "--top-k-features", type=int, default=15,
        help="Number of top features to select (default: 15). Use 0 to keep all.",
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Skip ADASYN class balancing.",
    )
    parser.add_argument(
        "--non-iid", action="store_true",
        help="Use non-IID data distribution.",
    )
    parser.add_argument(
        "--sample-n", type=int, default=None,
        help="Subsample the raw data to N rows before processing.",
    )
    parser.add_argument(
        "--keep-classes", nargs="+", default=None,
        help="Keep only these target classes.",
    )
    parser.add_argument(
        "-o", "--output-dir", default="poisoned_output",
        help="Directory for output CSVs (default: poisoned_output/).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args(argv)

    if not args.generate and not args.csv:
        parser.error("Provide CSV files or use --generate / -g.")

    from poison_fl.pipeline import run_pipeline

    result = run_pipeline(
        csv_paths=args.csv or None,
        target_col=args.target_col,
        n_clients=args.n_clients,
        poison_fraction=args.poison_fraction,
        noise_std=args.noise_std,
        top_k_features=args.top_k_features or None,
        balance=not args.no_balance,
        iid=not args.non_iid,
        sample_n=args.sample_n,
        keep_classes=args.keep_classes,
        output_dir=args.output_dir,
        random_state=args.seed,
        generate=args.generate,
        generate_n=args.generate_n,
        generate_classes=args.generate_classes,
        imbalance_ratio=args.imbalance_ratio,
    )

    n_total = sum(len(c["y"]) for c in result["clients"])
    n_poisoned = sum(c["poison_mask"].sum() for c in result["clients"])

    print(f"Done. {args.n_clients} clients, {n_total} total samples, "
          f"{n_poisoned} poisoned ({n_poisoned/n_total:.1%}).")
    print(f"Output saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
