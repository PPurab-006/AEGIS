#!/usr/bin/env python3
"""
Research 2 — Build Expanded Dataset from 42 Eligible Sweep Flights.

Constructs backward-lagged features and forward-looking failure window labels
across horizons K in {2, 3, 5} frames from the 42 eligible sweep flights.

Hardened Eligibility:
  48 original sweep flights - 6 ineligible = 42 eligible flights.
  Excluded flights:
    - sweep_G_B_R1
    - sweep_G_H_R1, sweep_G_H_R2, sweep_G_H_R3 (G_H cell entirely excluded)
    - sweep_M_H_R2, sweep_M_H_R3

Inputs:
  data/processed/expanded_eligible_flights.txt
  /home/purab/Purab/Projects/ROS/results/datasets/<flight_name>/

Outputs:
  data/processed/expanded_flight_manifest.csv
  data/processed/expanded_flight_split.csv
  data/processed/expanded_frames_v2_k5.csv
  data/processed/expanded_frames_v2_k3.csv
  data/processed/expanded_frames_v2_k2.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "eis_yaw_rate_deg",   # |omega_z|, absolute body yaw rate in deg/s
    "feature_vel_mean",   # KLT optical flow magnitude in px/frame
    "is_r_frame",         # binary EIS yaw-rate gate bypass flag (> 15 deg/s)
]

FEATURE_LAGS = [0, 1, 2, 3, 5]

META_OUT = [
    "run_dir",
    "family",
    "gt_discontinuity_flag",
    "is_failure_current_frame",
    "is_failure_within_next_k",
]

FEATURE_COLS = [f"{col}_lag{lag}" for col in BASE_FEATURES for lag in FEATURE_LAGS]
FINAL_COLUMNS = META_OUT + FEATURE_COLS

INELIGIBLE_FLIGHTS = {
    "sweep_G_B_R1",
    "sweep_G_H_R1",
    "sweep_G_H_R2",
    "sweep_G_H_R3",
    "sweep_M_H_R2",
    "sweep_M_H_R3",
}


def load_eligible_manifest(manifest_path: Path) -> list[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        flights = [line.strip() for line in f if line.strip()]

    # Sanity checks
    if len(flights) != 42:
        raise ValueError(f"Expected exactly 42 eligible flights in {manifest_path}, got {len(flights)}")
    if len(set(flights)) != 42:
        raise ValueError(f"Duplicate flight entries found in {manifest_path}")

    for f in flights:
        if f in INELIGIBLE_FLIGHTS or "G_H" in f:
            raise ValueError(f"Ineligible flight {f} detected in {manifest_path}")

    return flights


def process_flight(
    flight: str,
    data_dir: Path,
    k_variants: list[int] = [2, 3, 5],
) -> tuple[dict, dict[int, pd.DataFrame]]:
    """Process a single flight: active window filtering, base feature extraction,

    lagging, and forward-looking failure window labels for K in {2, 3, 5}.
    """
    gt_path = data_dir / flight / "dataset_gt.csv"
    raw_path = data_dir / flight / "raw_vo.csv"

    if not gt_path.exists() or not raw_path.exists():
        raise FileNotFoundError(f"Missing data for flight {flight}: gt={gt_path.exists()}, raw={raw_path.exists()}")

    gt = pd.read_csv(gt_path)
    raw = pd.read_csv(raw_path)

    # Required columns check
    for col in ["pos_z", "timestamp_total_sec"]:
        if col not in gt.columns:
            raise ValueError(f"{flight}: dataset_gt.csv missing '{col}'")
    for col in BASE_FEATURES + ["num_inliers_pose", "timestamp_total_sec"]:
        if col not in raw.columns:
            raise ValueError(f"{flight}: raw_vo.csv missing '{col}'")

    # Active window: gt.pos_z >= 2.0
    active_idx = gt["pos_z"].astype(float) >= 2.0
    if not active_idx.any():
        raise ValueError(f"{flight}: no ground-truth samples with pos_z >= 2.0")

    t0 = gt.loc[active_idx, "timestamp_total_sec"].astype(float).min()
    t1 = gt.loc[active_idx, "timestamp_total_sec"].astype(float).max()

    raw_t = raw["timestamp_total_sec"].astype(float)
    in_window = (raw_t >= t0) & (raw_t <= t1)
    df = raw.loc[in_window].copy().reset_index(drop=True)

    n_active = len(df)
    if n_active <= max(FEATURE_LAGS) + max(k_variants):
        raise ValueError(f"{flight}: insufficient active frames ({n_active})")

    # Raw failure label
    df["is_failure"] = (df["num_inliers_pose"].astype(int) < 8).astype(int)
    n_failures = int(df["is_failure"].sum())

    # Family derivation (cell): e.g. sweep_A_B_R1 -> A_B
    parts = flight.split("_")
    family = f"{parts[1]}_{parts[2]}"

    df["run_dir"] = flight
    df["family"] = family
    df["gt_discontinuity_flag"] = False

    manifest_row = {
        "family": family,
        "run_dir": flight,
        "n_active_frames": n_active,
        "n_failure_frames": n_failures,
        "failure_rate_pct": round(100.0 * n_failures / n_active, 2),
        "gt_discontinuity_flag": False,
    }

    # Base backward lagged features
    for col in BASE_FEATURES:
        for lag in FEATURE_LAGS:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df["is_failure_current_frame"] = df["is_failure"].astype(int)

    k_dfs = {}
    for k in k_variants:
        df_k = df.copy()
        forward_shifts = pd.concat([df_k["is_failure"].shift(-step) for step in range(k + 1)], axis=1)
        df_k["is_failure_within_next_k"] = forward_shifts.max(axis=1)
        df_k.loc[forward_shifts.isna().any(axis=1), "is_failure_within_next_k"] = np.nan

        # Intra-flight boundary drops: first 5, last K
        df_valid = df_k.dropna(subset=FEATURE_COLS + ["is_failure_within_next_k"]).copy().reset_index(drop=True)

        expected_rows = n_active - (max(FEATURE_LAGS) + k)
        if len(df_valid) != expected_rows:
            raise ValueError(f"{flight} (K={k}): expected {expected_rows} rows, got {len(df_valid)}")

        df_valid["is_failure_current_frame"] = df_valid["is_failure_current_frame"].astype(int)
        df_valid["is_failure_within_next_k"] = df_valid["is_failure_within_next_k"].astype(int)
        for lag in FEATURE_LAGS:
            df_valid[f"is_r_frame_lag{lag}"] = df_valid[f"is_r_frame_lag{lag}"].astype(int)

        k_dfs[k] = df_valid[FINAL_COLUMNS].copy()

    return manifest_row, k_dfs


def stratified_flight_split(
    manifest: pd.DataFrame,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    stratify_col: str = "family",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Flight-level stratified split preserving seed 42 and stratified group handling."""
    rng = np.random.default_rng(random_seed)
    manifest = manifest.copy()
    manifest["split"] = "train"

    for group_val, group_df in manifest.groupby(stratify_col):
        flights = np.array(group_df["run_dir"].unique().tolist())
        rng.shuffle(flights)
        n = len(flights)
        n_test = max(1, round(n * test_fraction)) if n >= 3 else (1 if n > 1 else 0)
        n_val = max(1, round(n * val_fraction)) if n >= 3 else 0
        n_test = min(n_test, n - 1) if n > 1 else 0
        n_val = min(n_val, n - n_test - 1) if (n - n_test) > 1 else 0

        test_flights = set(flights[:n_test])
        val_flights = set(flights[n_test : n_test + n_val])

        manifest.loc[manifest["run_dir"].isin(test_flights), "split"] = "test"
        manifest.loc[manifest["run_dir"].isin(val_flights), "split"] = "val"

    return manifest


def main():
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    data_dir = Path("/home/purab/Purab/Projects/ROS/results/datasets")
    manifest_txt = processed_dir / "expanded_eligible_flights.txt"

    print("=================================================================")
    print("Research 2 — Building Expanded Datasets (42 Eligible Flights)")
    print("=================================================================")

    flights = load_eligible_manifest(manifest_txt)
    print(f"Loaded {len(flights)} eligible flight IDs from {manifest_txt.name}")

    manifest_rows = []
    k_accumulators = {2: [], 3: [], 5: []}

    for idx, flight in enumerate(flights, 1):
        m_row, k_dfs = process_flight(flight, data_dir, k_variants=[2, 3, 5])
        manifest_rows.append(m_row)
        for k in [2, 3, 5]:
            k_accumulators[k].append(k_dfs[k])

    df_manifest = pd.DataFrame(manifest_rows)
    df_split = stratified_flight_split(
        df_manifest,
        test_fraction=0.2,
        val_fraction=0.2,
        stratify_col="family",
        random_seed=42,
    )

    # Save manifest and split
    out_manifest = processed_dir / "expanded_flight_manifest.csv"
    out_split = processed_dir / "expanded_flight_split.csv"
    df_manifest.to_csv(out_manifest, index=False)
    df_split.to_csv(out_split, index=False)
    print(f"Saved: {out_manifest.name}")
    print(f"Saved: {out_split.name}")

    # Build and save concatenated datasets
    for k in [2, 3, 5]:
        df_k = pd.concat(k_accumulators[k], ignore_index=True)
        out_k = processed_dir / f"expanded_frames_v2_k{k}.csv"
        df_k.to_csv(out_k, index=False)
        pos = int(df_k["is_failure_within_next_k"].sum())
        tot = len(df_k)
        pct = 100.0 * pos / tot
        print(f"Saved: {out_k.name:32s} | Rows: {tot:5d} | Pos: {pos:5d} ({pct:.2f}%)")

    # Split diagnostics
    print("\n--- Flight-Level Split Summary (Seed 42) ---")
    split_map = dict(zip(df_split["run_dir"], df_split["split"]))
    df_k5 = pd.concat(k_accumulators[5], ignore_index=True)
    df_k5["split"] = df_k5["run_dir"].map(split_map)

    for s in ["train", "val", "test"]:
        sub = df_k5[df_k5["split"] == s]
        n_fl = sub["run_dir"].nunique()
        n_rows = len(sub)
        n_pos = int(sub["is_failure_within_next_k"].sum())
        pct = 100.0 * n_pos / n_rows if n_rows > 0 else 0
        print(f"  {s.upper():5s}: {n_fl:2d} flights | {n_rows:5d} rows | {n_pos:5d} positives ({pct:.2f}%)")

    # Sanity checks
    train_flights = set(df_split[df_split["split"] == "train"]["run_dir"])
    val_flights = set(df_split[df_split["split"] == "val"]["run_dir"])
    test_flights = set(df_split[df_split["split"] == "test"]["run_dir"])

    assert len(train_flights.intersection(val_flights)) == 0, "Train and Val flight overlap!"
    assert len(train_flights.intersection(test_flights)) == 0, "Train and Test flight overlap!"
    assert len(val_flights.intersection(test_flights)) == 0, "Val and Test flight overlap!"
    assert len(train_flights) + len(val_flights) + len(test_flights) == 42, "Flight count mismatch!"

    print("\n[OK] Zero flight leakage across splits. All integrity assertions passed.")


if __name__ == "__main__":
    main()
