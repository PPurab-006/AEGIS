#!/usr/bin/env python3
"""
Phase 0, Step 2 — Build the labeled frame-level dataset.

Reads the RAW-condition telemetry fetched by 01_fetch_data.py, applies
the active-window filter, computes the failure label per the locked
definition in phase0_config.yaml, and writes:

  - data/processed/frames_labeled.csv   (one row per active-window frame)
  - data/processed/flight_manifest.csv  (one row per flight, for splitting)
  - data/processed/phase0_validation_report.md

This script does NOT do any feature engineering (that's Phase 1) and
does NOT do any train/test splitting (that's the next script). Its only
job is: fetch -> label -> validate against expected totals.

Usage:
    python scripts/02_build_labels.py --config configs/phase0_config.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def label_one_flight(
    run_dir: str,
    family: str,
    raw_csv_path: Path,
    gt_csv_path: Path,
    failure_rule_threshold: int,
    active_window_min_z: float,
    gt_discontinuity_flights: set,
) -> pd.DataFrame:
    """Load one flight's raw_vo.csv + dataset_gt.csv, apply active-window
    filter, compute the failure label, and return a tidy per-frame frame.
    """
    raw = pd.read_csv(raw_csv_path)
    gt = pd.read_csv(gt_csv_path)

    if "num_inliers_pose" not in raw.columns:
        raise ValueError(
            f"{run_dir}: raw_vo.csv is missing 'num_inliers_pose' column — "
            f"cannot compute failure label. Check pipeline version."
        )
    if "timestamp_total_sec" not in raw.columns or "timestamp_total_sec" not in gt.columns:
        raise ValueError(
            f"{run_dir}: missing 'timestamp_total_sec' column in raw_vo.csv "
            f"or dataset_gt.csv — cannot align active window."
        )
    if "pos_z" not in gt.columns:
        raise ValueError(f"{run_dir}: dataset_gt.csv missing 'pos_z' column.")

    # Determine active window [t0, t1] from ground truth (pos_z >= threshold)
    gt_t = gt["timestamp_total_sec"].astype(float).values
    gt_z = gt["pos_z"].astype(float).values
    active_idx = gt_z >= active_window_min_z

    if active_idx.sum() == 0:
        raise ValueError(
            f"{run_dir}: no ground-truth samples with pos_z >= "
            f"{active_window_min_z} — flight may never have reached "
            f"altitude, or GT data is malformed."
        )

    t0 = gt_t[active_idx].min()
    t1 = gt_t[active_idx].max()

    raw_t = raw["timestamp_total_sec"].astype(float).values
    in_window = (raw_t >= t0) & (raw_t <= t1)
    df = raw.loc[in_window].copy()

    df["is_failure"] = (df["num_inliers_pose"] < failure_rule_threshold).astype(int)
    df["run_dir"] = run_dir
    df["family"] = family
    df["gt_discontinuity_flag"] = run_dir in gt_discontinuity_flights

    return df


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
    raw_data_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_dir.is_dir() or not any(raw_data_dir.iterdir()):
        print(
            f"ERROR: {raw_data_dir} is empty or missing. "
            f"Run scripts/01_fetch_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    labeling_cfg = config["labeling"]
    failure_threshold = int(labeling_cfg["failure_definition"]["rule"].split("<")[1].strip())
    active_window_min_z = float(labeling_cfg["active_window"]["rule"].split(">=")[1].strip())
    gt_discontinuity_flights = set(config["split"]["gt_discontinuity_flights_flagged"])
    run_directories = config["run_directories"]

    print(f"Failure rule       : num_inliers_pose < {failure_threshold}")
    print(f"Active window rule : pos_z >= {active_window_min_z}")
    print(f"Families included  : {list(run_directories.keys())}")
    print()

    all_frames = []
    flight_summaries = []

    for family, run_dirs in run_directories.items():
        for run_dir in run_dirs:
            flight_dir = raw_data_dir / run_dir
            raw_csv = flight_dir / "raw_vo.csv"
            gt_csv = flight_dir / "dataset_gt.csv"

            if not raw_csv.exists() or not gt_csv.exists():
                print(f"  [SKIP] {run_dir}: missing raw_vo.csv or dataset_gt.csv")
                continue

            df = label_one_flight(
                run_dir=run_dir,
                family=family,
                raw_csv_path=raw_csv,
                gt_csv_path=gt_csv,
                failure_rule_threshold=failure_threshold,
                active_window_min_z=active_window_min_z,
                gt_discontinuity_flights=gt_discontinuity_flights,
            )
            all_frames.append(df)

            n_frames = len(df)
            n_fail = int(df["is_failure"].sum())
            rate = 100.0 * n_fail / n_frames if n_frames else 0.0
            flight_summaries.append(
                {
                    "family": family,
                    "run_dir": run_dir,
                    "n_active_frames": n_frames,
                    "n_failure_frames": n_fail,
                    "failure_rate_pct": round(rate, 2),
                    "gt_discontinuity_flag": run_dir in gt_discontinuity_flights,
                }
            )
            flag = " [GT-DISCONTINUITY]" if run_dir in gt_discontinuity_flights else ""
            print(f"  [OK] {family:6s} {run_dir:24s} n={n_frames:5d}  fail={n_fail:4d}  rate={rate:5.2f}%{flag}")

    if not all_frames:
        print("ERROR: no flights were successfully labeled.", file=sys.stderr)
        sys.exit(1)

    frames_df = pd.concat(all_frames, ignore_index=True)
    flight_manifest = pd.DataFrame(flight_summaries)

    frames_out_path = processed_dir / "frames_labeled.csv"
    manifest_out_path = processed_dir / "flight_manifest.csv"
    frames_df.to_csv(frames_out_path, index=False)
    flight_manifest.to_csv(manifest_out_path, index=False)

    # ---- Validation against expected totals from the audit ----
    expected = config["expected_totals"]
    tolerance_pct = float(expected["tolerance_pct"])

    n_flights_actual = flight_manifest["run_dir"].nunique()
    n_frames_actual = len(frames_df)
    n_fail_actual = int(frames_df["is_failure"].sum())
    rate_actual = 100.0 * n_fail_actual / n_frames_actual if n_frames_actual else 0.0

    def pct_diff(actual, expected_val):
        if expected_val == 0:
            return 0.0
        return 100.0 * abs(actual - expected_val) / expected_val

    checks = [
        ("n_flights", n_flights_actual, expected["n_flights"]),
        ("active_frames_total", n_frames_actual, expected["active_frames_total_approx"]),
        ("active_failure_frames", n_fail_actual, expected["active_failure_frames_approx"]),
        ("active_failure_rate_pct", round(rate_actual, 2), expected["active_failure_rate_approx_pct"]),
    ]

    print("\n--- Validation against Phase 0 audit expectations ---")
    any_out_of_tolerance = False
    report_lines = [
        "# Phase 0 Validation Report\n",
        f"Failure rule: `num_inliers_pose < {failure_threshold}`  \n",
        f"Active window rule: `pos_z >= {active_window_min_z}`  \n",
        f"Ground truth condition: `{labeling_cfg['ground_truth_condition']}`  \n",
        f"Excluded: `{labeling_cfg['excluded_condition']}` "
        f"({labeling_cfg['excluded_condition_reason'].strip()[:80]}...)  \n",
        "\n## Checks\n",
        "| Metric | Actual | Expected (audit) | Diff % | Status |",
        "|---|---|---|---|---|",
    ]
    for name, actual, expected_val in checks:
        diff = pct_diff(actual, expected_val)
        status = "OK" if diff <= tolerance_pct else "REVIEW NEEDED"
        if diff > tolerance_pct:
            any_out_of_tolerance = True
        print(f"  {name:28s} actual={actual:>10}  expected~={expected_val:>10}  diff={diff:5.1f}%  [{status}]")
        report_lines.append(f"| {name} | {actual} | {expected_val} | {diff:.1f}% | {status} |")

    report_lines.append("\n## Per-family breakdown\n")
    fam_agg = (
        flight_manifest.groupby("family")
        .agg(
            n_flights=("run_dir", "nunique"),
            n_frames=("n_active_frames", "sum"),
            n_failures=("n_failure_frames", "sum"),
        )
        .reset_index()
    )
    fam_agg["failure_rate_pct"] = round(100.0 * fam_agg["n_failures"] / fam_agg["n_frames"], 2)
    report_lines.append("| Family | Flights | Frames | Failures | Rate % |")
    report_lines.append("|---|---|---|---|---|")
    for _, row in fam_agg.iterrows():
        report_lines.append(
            f"| {row['family']} | {row['n_flights']} | {row['n_frames']} | "
            f"{row['n_failures']} | {row['failure_rate_pct']} |"
        )

    report_lines.append("\n## GT-discontinuity flights (flagged, not excluded)\n")
    disc_rows = flight_manifest[flight_manifest["gt_discontinuity_flag"]]
    if len(disc_rows):
        for _, row in disc_rows.iterrows():
            report_lines.append(f"- `{row['run_dir']}` ({row['family']}): {row['n_failure_frames']} failures / {row['n_active_frames']} frames")
    else:
        report_lines.append("- None found in fetched data (check config if this is unexpected).")

    validation_report_path = processed_dir / "phase0_validation_report.md"
    with open(validation_report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nLabeled frames written to   : {frames_out_path}")
    print(f"Flight manifest written to  : {manifest_out_path}")
    print(f"Validation report written to: {validation_report_path}")

    if any_out_of_tolerance:
        print(
            "\nWARNING: one or more totals deviated from the audit's expected "
            "values by more than the configured tolerance. This does NOT "
            "necessarily mean something is wrong (e.g. minor CSV row-count "
            "differences are normal) but review phase0_validation_report.md "
            "before treating this dataset as final.",
            file=sys.stderr,
        )
        sys.exit(2)

    print("\nAll checks within tolerance. Phase 0 data assembly complete.")


if __name__ == "__main__":
    main()
