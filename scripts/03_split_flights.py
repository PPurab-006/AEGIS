#!/usr/bin/env python3
"""
Phase 0, Step 3 — Flight-level train/val/test split.

CRITICAL: This split happens at the FLIGHT level, never the frame level.
Frames within a single flight are temporally and spatially correlated;
splitting at the frame level would leak information between train and
test sets and produce an inflated, untrustworthy result. This is the
single most important methodological safeguard in Phase 0 — do not
bypass it, even for quick experiments.

Reads data/processed/flight_manifest.csv (from 02_build_labels.py) and
writes data/processed/flight_split.csv assigning each flight to
train/val/test. This split is fixed by random_seed in the config — it
should be computed ONCE and then treated as immutable for the rest of
the project. If you need to change split logic, do so deliberately and
document why in the config, not by re-running with a different seed
until you get numbers you like.

Usage:
    python scripts/03_split_flights.py --config configs/phase0_config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stratified_flight_split(
    manifest: pd.DataFrame,
    test_fraction: float,
    val_fraction: float,
    stratify_col: str,
    random_seed: int,
) -> pd.DataFrame:
    """Assign each flight to train/val/test, stratified by `stratify_col`
    (typically family) so the held-out sets aren't accidentally dominated
    by a single family's failure-rate profile.
    """
    rng = np.random.default_rng(random_seed)
    manifest = manifest.copy()
    manifest["split"] = "train"

    for group_val, group_df in manifest.groupby(stratify_col):
        flights = np.array(group_df["run_dir"].unique().tolist())
        rng.shuffle(flights)
        n = len(flights)
        n_test = max(1, round(n * test_fraction)) if n >= 3 else (1 if n > 1 else 0)
        n_val = max(1, round(n * val_fraction)) if n >= 3 else 0
        # guard against over-allocating on very small groups
        n_test = min(n_test, n - 1) if n > 1 else 0
        n_val = min(n_val, n - n_test - 1) if (n - n_test) > 1 else 0

        test_flights = set(flights[:n_test])
        val_flights = set(flights[n_test : n_test + n_val])

        manifest.loc[manifest["run_dir"].isin(test_flights), "split"] = "test"
        manifest.loc[manifest["run_dir"].isin(val_flights), "split"] = "val"

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "phase0_config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    manifest_path = processed_dir / "flight_manifest.csv"

    if not manifest_path.exists():
        print(
            f"ERROR: {manifest_path} not found. Run scripts/02_build_labels.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    split_cfg = config["split"]

    split_manifest = stratified_flight_split(
        manifest=manifest,
        test_fraction=float(split_cfg["test_fraction"]),
        val_fraction=float(split_cfg["val_fraction"]),
        stratify_col=split_cfg["stratify_by"],
        random_seed=int(split_cfg["random_seed"]),
    )

    out_path = processed_dir / "flight_split.csv"
    split_manifest.to_csv(out_path, index=False)

    print("Flight-level split (stratified by family):\n")
    summary = split_manifest.groupby(["family", "split"]).size().unstack(fill_value=0)
    print(summary)

    print("\nOverall split counts:")
    print(split_manifest["split"].value_counts())

    # Report failure-rate balance across splits — a sanity check that the
    # held-out set isn't accidentally all-low-failure-rate flights
    print("\nFailure rate by split (sanity check — should be roughly comparable):")
    rate_by_split = split_manifest.groupby("split").apply(
        lambda g: 100.0 * g["n_failure_frames"].sum() / g["n_active_frames"].sum()
    )
    print(rate_by_split.round(2))

    # Flag where GT-discontinuity flights landed, for transparency
    disc = split_manifest[split_manifest["gt_discontinuity_flag"]]
    if len(disc):
        print("\nGT-discontinuity flights landed in:")
        for _, row in disc.iterrows():
            print(f"  {row['run_dir']} ({row['family']}) -> {row['split']}")

    print(f"\nSplit written to {out_path}")
    print(
        "\nThis split is now LOCKED for the rest of the project. Do not "
        "re-run with a different seed to change which flights land where — "
        "if the split needs to change, document why in phase0_config.yaml."
    )


if __name__ == "__main__":
    main()
