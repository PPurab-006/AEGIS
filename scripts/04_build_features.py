#!/usr/bin/env python3
"""
Phase 1, Step 4 — Build the v1 feature table.

Reads the Phase 0 labeled frame dataset (data/processed/frames_labeled.csv)
and constructs the locked v1 feature set for Research 2's learned failure
predictor. Writes:

  data/processed/frames_with_features.csv  — frames_labeled.csv + v1 features
  data/processed/phase1_feature_report.md  — regression stats, feature
                                             summaries, other-column inspection

=== Locked v1 Feature Set ===

1. eis_yaw_rate_deg   — |omega_z|, absolute body yaw rate in deg/s.
   Source: raw_vo.csv column 'eis_yaw_rate_deg', already in frames_labeled.csv.
   Computation (R1): abs(rotvec_body[2] / dt) where rotvec is the body-frame
   rotation vector between consecutive SLERP-interpolated GT attitude samples.

2. feature_vel_mean   — Mean KLT optical-flow magnitude in pixels/frame.
   Source: raw_vo.csv column 'feature_vel_mean', already in frames_labeled.csv.
   KNOWN LIMITATION: In RAW mode the EIS derotation stage is never applied,
   so this optical-flow signal conflates rotational and translational apparent
   motion. It is a proxy for the velocity that drives scene parallax, not a
   clean translational velocity signal. Do not interpret it as such. The
   vel_residual feature below is a crude attempt to separate the two, but it
   is also not precise -- see its own docstring.

3. is_r_frame   — Binary flag: 1 if eis_yaw_rate_deg > 15.0 deg/s (the EIS
   gate threshold), 0 otherwise. Marks frames that would have triggered EIS
   bypass in the EIS-GATED condition. Source: raw_vo.csv 'is_r_frame' column.

4. vel_residual   — Engineered feature. Residual from a linear regression of
   feature_vel_mean ~ eis_yaw_rate_deg, fit on TRAIN-SPLIT FLIGHTS ONLY.
   Intended as a crude proxy for translational motion not explained by
   rotation alone. Not a precise optical-flow decomposition -- the relationship
   between yaw rate and apparent flow depends on scene depth and geometry
   in ways a global linear model cannot capture. Treat as a weak signal.
   The regression coefficients and R^2 are reported in phase1_feature_report.md.

=== Explicitly Dropped ===

eis_gate_scale: Always 1.0 in the RAW condition (RAW mode never invokes the
gated/scaled code path in eis_preprocessor.py). Contains zero discriminative
information. Absent from the output feature table by design.

=== Inputs ===

  data/processed/frames_labeled.csv   — Phase 0 output; all v1 source
                                        columns are already present here
  data/processed/flight_split.csv     — for train-only regression fit

Usage:
    python scripts/04_build_features.py --config configs/phase0_config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Locked v1 feature column names (sourced from frames_labeled.csv)
# ---------------------------------------------------------------------------
V1_FEATURE_COLS = [
    "eis_yaw_rate_deg",   # |omega_z|
    "feature_vel_mean",   # translational-velocity proxy (conflated in RAW)
    "is_r_frame",         # binary yaw-rate gate flag
]
ENGINEERED_FEATURE = "vel_residual"
DROP_COL = "eis_gate_scale"  # always 1.0 in RAW -- excluded by design

# Columns that are metadata / label / index -- NOT "other features" for inspection
META_COLS = {
    "frame_idx", "timestamp_sec", "timestamp_nanosec", "timestamp_total_sec",
    "is_failure", "run_dir", "family", "gt_discontinuity_flag",
    # v1 feature cols
    "eis_yaw_rate_deg", "feature_vel_mean", "is_r_frame", "vel_residual",
    # the drop col
    "eis_gate_scale",
    # gt position / orientation (not VO telemetry features)
    "pos_x", "pos_y", "pos_z",
    "rot_x", "rot_y", "rot_z", "rot_w",
}

# Columns known to be always 0 / constant in RAW mode
RAW_CONSTANT_COLS = {"eis_crop_pct", "eis_warp_deg", "eis_cum_warp_deg",
                     "num_pending", "promotions_this_frame"}


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fit_vel_residual_regression(df_train):
    """Fit feature_vel_mean ~ eis_yaw_rate_deg on training frames only.
    Returns (intercept, slope, r_squared).
    """
    x = df_train["eis_yaw_rate_deg"].astype(float).values
    y = df_train["feature_vel_mean"].astype(float).values

    A = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = float(coeffs[0]), float(coeffs[1])

    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return intercept, slope, r_squared


def feature_stats(series, label_series):
    """Compute summary stats overall and split by is_failure label."""
    vals = series.astype(float)
    fail_mask = label_series.astype(int) == 1
    ok_mask = ~fail_mask

    def safe_stats(s):
        if len(s) == 0:
            return dict(n=0, mean=float("nan"), std=float("nan"),
                        min=float("nan"), p25=float("nan"), p50=float("nan"),
                        p75=float("nan"), max=float("nan"))
        return dict(
            n=len(s),
            mean=round(float(s.mean()), 6),
            std=round(float(s.std()), 6),
            min=round(float(s.min()), 6),
            p25=round(float(s.quantile(0.25)), 6),
            p50=round(float(s.median()), 6),
            p75=round(float(s.quantile(0.75)), 6),
            max=round(float(s.max()), 6),
        )

    return {
        "overall": safe_stats(vals),
        "is_failure=0": safe_stats(vals[ok_mask]),
        "is_failure=1": safe_stats(vals[fail_mask]),
    }


def fmt_stats(stats_dict):
    """Format a feature_stats dict as a compact markdown table."""
    groups = ["overall", "is_failure=0", "is_failure=1"]
    header = "| Stat | " + " | ".join(groups) + " |"
    sep    = "|---" * (len(groups) + 1) + "|"
    rows = []
    for stat in ["n", "mean", "std", "min", "p25", "p50", "p75", "max"]:
        row_vals = [str(stats_dict[g].get(stat, "")) for g in groups]
        rows.append("| " + stat + " | " + " | ".join(row_vals) + " |")
    return "\n".join([header, sep] + rows)


def inspect_other_col(col, df):
    """Compute inspection stats for a non-v1 column."""
    s = df[col]
    dtype_str = str(s.dtype)
    fail_mask = df["is_failure"].astype(int) == 1

    def smean(series):
        if len(series) == 0 or series.isna().all():
            return float("nan")
        try:
            return round(float(series.astype(float).mean()), 5)
        except Exception:
            return float("nan")

    n_unique = s.nunique()
    is_constant = (n_unique == 1)

    try:
        all_zero = bool((s.astype(float) == 0).all())
    except Exception:
        all_zero = False

    flag_const = is_constant or all_zero or (col in RAW_CONSTANT_COLS)

    def sfloat(v):
        try:
            return round(float(v), 5)
        except Exception:
            return "N/A"

    return {
        "col": col,
        "dtype": dtype_str,
        "n_unique": n_unique,
        "is_constant": is_constant,
        "all_zero_in_raw": all_zero or col in RAW_CONSTANT_COLS,
        "flag_const": flag_const,
        "min": sfloat(s.astype(float).min()),
        "max": sfloat(s.astype(float).max()),
        "mean": smean(s),
        "std": sfloat(s.astype(float).std()),
        "mean_fail0": smean(s[~fail_mask]),
        "mean_fail1": smean(s[fail_mask]),
    }


def build_report(df, intercept, slope, r_squared,
                 n_train_frames, n_train_flights, other_col_stats):
    lines = []

    lines.append("# Phase 1 Feature Report — Research 2")
    lines.append("")
    lines.append("Ground-truth condition: `RAW`  ")
    lines.append("Failure rule: `num_inliers_pose < 8`  ")
    lines.append("Active-window rule: `pos_z >= 2.0`  ")
    lines.append("")

    lines.append("## v1 Feature Set")
    lines.append("")
    lines.append("| # | Column | Source | Notes |")
    lines.append("|---|---|---|---|")
    lines.append("| 1 | `eis_yaw_rate_deg` | `raw_vo.csv` | |omega_z|, deg/s |")
    lines.append("| 2 | `feature_vel_mean` | `raw_vo.csv` | Mean KLT flow, px/frame. **Conflated in RAW mode** (rotation + translation mixed) |")
    lines.append("| 3 | `is_r_frame` | `raw_vo.csv` | Binary: 1 if eis_yaw_rate_deg > 15.0 deg/s |")
    lines.append("| 4 | `vel_residual` | Engineered | feature_vel_mean minus (intercept + slope x eis_yaw_rate_deg), fit on train only |")
    lines.append("")
    lines.append("> **Explicitly absent:** `eis_gate_scale` -- always 1.0 in RAW condition, zero discriminative information.")
    lines.append("")

    lines.append("## vel_residual Regression Fit")
    lines.append("")
    lines.append("Model: `feature_vel_mean ~ intercept + slope x eis_yaw_rate_deg`  ")
    lines.append(f"Fit on: **{n_train_flights} train-split flights** ({n_train_frames:,} active-window frames)  ")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| intercept | {intercept:.6f} px/frame |")
    lines.append(f"| slope | {slope:.6f} px/frame per deg/s |")
    lines.append(f"| R2 (train) | {r_squared:.6f} |")
    lines.append("")
    r2_pct = r_squared * 100
    lines.append(
        f"Interpretation: rotation (eis_yaw_rate_deg) explains **{r2_pct:.1f}%** of the "
        f"variance in feature_vel_mean on the training set. `vel_residual` is the unexplained "
        f"remainder -- a crude proxy for translational apparent motion not accounted for by "
        f"yaw rate alone. This is not a precise flow decomposition: scene depth, feature "
        f"distribution, and non-yaw rotations all affect the relationship in ways a global "
        f"linear model cannot capture."
    )
    lines.append("")

    lines.append("## v1 Feature Summary Statistics")
    lines.append("")
    lines.append("Active-window frames only. All 33 flights (train + val + test).")
    lines.append("")

    v1_cols_report = ["eis_yaw_rate_deg", "feature_vel_mean", "is_r_frame", "vel_residual"]
    for col in v1_cols_report:
        lines.append(f"### `{col}`")
        stats = feature_stats(df[col], df["is_failure"])
        lines.append(fmt_stats(stats))
        lines.append("")

    lines.append("## Label Distribution")
    lines.append("")
    n_total = len(df)
    n_fail = int(df["is_failure"].sum())
    n_ok = n_total - n_fail
    lines.append("| | Frames | Fraction |")
    lines.append("|---|---|---|")
    lines.append(f"| is_failure=0 | {n_ok:,} | {n_ok/n_total:.4f} |")
    lines.append(f"| is_failure=1 | {n_fail:,} | {n_fail/n_total:.4f} |")
    lines.append(f"| **Total** | **{n_total:,}** | |")
    lines.append("")

    lines.append("## Other Columns in frames_labeled.csv -- Inspection Only")
    lines.append("")
    lines.append(
        "These columns are present in `frames_labeled.csv` (carried over from `raw_vo.csv`) "
        "but are **not** part of the v1 feature set. Reported here for information only; "
        "no decision to include them has been made. Columns marked [CONST/ZERO] are constant "
        "or zero in RAW mode and carry no information."
    )
    lines.append("")
    lines.append("| Column | dtype | n_unique | const/zero | min | max | mean | std | mean(fail=0) | mean(fail=1) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for info in other_col_stats:
        flag = " [CONST/ZERO]" if info["flag_const"] else ""
        lines.append(
            f"| `{info['col']}`{flag} | {info['dtype']} | {info['n_unique']} | "
            f"{'yes' if info['flag_const'] else 'no'} | "
            f"{info['min']} | {info['max']} | {info['mean']} | {info['std']} | "
            f"{info['mean_fail0']} | {info['mean_fail1']} |"
        )

    lines.append("")
    lines.append(
        "> **Note on RAW-constant columns:** `eis_crop_pct`, `eis_warp_deg`, "
        "`eis_cum_warp_deg` are 0.0 in RAW mode (EIS derotation never applied). "
        "`num_pending` and `promotions_this_frame` are 0 because delayed triangulation "
        "is not active in RAW mode. These cannot carry predictive signal within the "
        "RAW-only dataset."
    )
    lines.append("")

    return "\n".join(lines)


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

    frames_path = processed_dir / "frames_labeled.csv"
    split_path  = processed_dir / "flight_split.csv"
    out_frames  = processed_dir / "frames_with_features.csv"
    out_report  = processed_dir / "phase1_feature_report.md"

    # ------------------------------------------------------------------
    # 1. Load Phase 0 outputs
    # ------------------------------------------------------------------
    print("Loading Phase 0 outputs...")

    if not frames_path.exists():
        print(f"\nERROR: {frames_path} not found -- run Phase 0 first.", file=sys.stderr)
        sys.exit(1)
    if not split_path.exists():
        print(f"\nERROR: {split_path} not found -- run Phase 0 first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(frames_path)
    df_split = pd.read_csv(split_path)

    print(f"  Loaded {len(df):,} labeled frames from {frames_path.name}")
    print(f"  Loaded split table: {len(df_split)} flights")

    # Sanity check: required columns present
    required = V1_FEATURE_COLS + ["is_failure", "run_dir", DROP_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: frames_labeled.csv is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Assert eis_gate_scale is constant (always 1.0)
    # ------------------------------------------------------------------
    gate_unique = df[DROP_COL].unique()
    if len(gate_unique) != 1 or float(gate_unique[0]) != 1.0:
        print(
            f"\nWARNING: {DROP_COL} has non-1.0 values: {gate_unique[:5]}\n"
            f"  This contradicts expected RAW-mode behaviour. Dropping anyway.",
            file=sys.stderr,
        )
    else:
        print(f"  Confirmed: {DROP_COL} is always 1.0 in RAW condition -- dropping.")

    # ------------------------------------------------------------------
    # 3. Build train-split frame mask for regression fitting
    # ------------------------------------------------------------------
    train_flights = set(df_split.loc[df_split["split"] == "train", "run_dir"].tolist())
    val_flights   = set(df_split.loc[df_split["split"] == "val",   "run_dir"].tolist())
    test_flights  = set(df_split.loc[df_split["split"] == "test",  "run_dir"].tolist())

    n_train_flights = len(train_flights)
    n_val_flights   = len(val_flights)
    n_test_flights  = len(test_flights)

    train_mask = df["run_dir"].isin(train_flights)
    df_train = df[train_mask].copy()
    n_train_frames = len(df_train)

    print(f"\nSplit summary:")
    print(f"  train: {n_train_flights} flights,  {n_train_frames:,} active frames")
    print(f"  val  : {n_val_flights} flights,   {len(df[df['run_dir'].isin(val_flights)]):,} active frames")
    print(f"  test : {n_test_flights} flights,  {len(df[df['run_dir'].isin(test_flights)]):,} active frames")

    # ------------------------------------------------------------------
    # 4. Fit vel_residual regression (TRAIN ONLY)
    # ------------------------------------------------------------------
    print(f"\nFitting vel_residual regression on train flights only ({n_train_frames:,} frames)...")
    intercept, slope, r_squared = fit_vel_residual_regression(df_train)
    print(f"  intercept  : {intercept:.6f} px/frame")
    print(f"  slope      : {slope:.6f} px/frame per deg/s")
    print(f"  R2 (train) : {r_squared:.6f}  ({r_squared*100:.1f}% variance explained by yaw rate)")

    # ------------------------------------------------------------------
    # 5. Apply regression to ALL frames, compute residual
    # ------------------------------------------------------------------
    yaw = df["eis_yaw_rate_deg"].astype(float).values
    vel = df["feature_vel_mean"].astype(float).values
    vel_pred = intercept + slope * yaw
    df["vel_residual"] = (vel - vel_pred).round(8)

    # ------------------------------------------------------------------
    # 6. Print v1 feature summary to console
    # ------------------------------------------------------------------
    print("\n--- v1 Feature Summary (all 33 flights, active window) ---")
    v1_cols_all = V1_FEATURE_COLS + [ENGINEERED_FEATURE]
    for col in v1_cols_all:
        s = df[col].astype(float)
        fail_mask = df["is_failure"].astype(int) == 1
        m_fail = s[fail_mask].mean()
        m_ok   = s[~fail_mask].mean()
        print(
            f"  {col:<22s}  mean={s.mean():>9.4f}  std={s.std():>9.4f}  "
            f"mean(fail=0)={m_ok:>9.4f}  mean(fail=1)={m_fail:>9.4f}"
        )

    # ------------------------------------------------------------------
    # 7. Inspect "other" columns
    # ------------------------------------------------------------------
    other_cols = sorted([
        c for c in df.columns
        if c not in META_COLS and c != DROP_COL
    ])
    print(f"\nInspecting {len(other_cols)} additional columns (not part of v1 feature set)...")

    other_col_stats = []
    for col in other_cols:
        info = inspect_other_col(col, df)
        flag = " [CONST/ZERO]" if info["flag_const"] else ""
        print(
            f"  {col:<35s}  mean(fail=0)={info['mean_fail0']:>10.5f}  "
            f"mean(fail=1)={info['mean_fail1']:>10.5f}{flag}"
        )
        other_col_stats.append(info)

    # ------------------------------------------------------------------
    # 8. Build output feature table (drop eis_gate_scale)
    # ------------------------------------------------------------------
    # Build ordered column list: keep existing order, insert vel_residual
    # after is_r_frame, remove DROP_COL
    base_cols = [c for c in df.columns if c != DROP_COL and c != "vel_residual"]
    out_cols_ordered = []
    for c in base_cols:
        out_cols_ordered.append(c)
        if c == "is_r_frame":
            out_cols_ordered.append("vel_residual")

    df_out = df[out_cols_ordered]
    df_out.to_csv(out_frames, index=False)

    print(f"\nFeature table written to {out_frames}")
    print(f"  Rows    : {len(df_out):,}")
    print(f"  Columns : {len(df_out.columns)}")
    assert DROP_COL not in df_out.columns, f"BUG: {DROP_COL} found in output!"
    assert "vel_residual" in df_out.columns, "BUG: vel_residual missing from output!"
    print(f"  Verified: '{DROP_COL}' absent  |  'vel_residual' present")

    # ------------------------------------------------------------------
    # 9. Write markdown report
    # ------------------------------------------------------------------
    report_md = build_report(
        df=df,
        intercept=intercept,
        slope=slope,
        r_squared=r_squared,
        n_train_frames=n_train_frames,
        n_train_flights=n_train_flights,
        other_col_stats=other_col_stats,
    )
    out_report.write_text(report_md, encoding="utf-8")
    print(f"Report written to {out_report}")

    print("\nDone. Phase 1 feature construction complete.")
    print("Next step: review phase1_feature_report.md before proceeding to Phase 2 modeling.")


if __name__ == "__main__":
    main()
