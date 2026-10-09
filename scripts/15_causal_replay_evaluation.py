#!/usr/bin/env python3
"""
Research 2 — Causal Replay & Latency / Lead-Time Evaluation.

Executes strictly causal, sample-by-sample replay across unseen held-out test flights.

Evaluates:
  1. Causal Operation & Numerical Integrity:
     Replays each test flight sample-by-sample in timestamp order.
     Zero future information in inference path.
  2. Computational Latency:
     Separately tracks Feature Construction, Scaler, MLP Inference, and End-to-End Latency.
     Reports mean, median, P95, P99, max, and compares against ~33.3 ms sampling period.
  3. Prediction Lead Time:
     Extracts all ground-truth failure episodes (num_inliers_pose < 8).
     For each episode, computes lead time: t_first_failure - t_prediction.
     Reports median, mean, P25, P75, min, max lead time.
  4. Categorization:
     - TRUE EARLY WARNING: Warning generated before failure onset.
     - POST-FAILURE DETECTION: Warning only generated after failure begins.
     - MISSED FAILURE: No warning generated for episode.
  5. Warning Stability:
     Tracks positive transitions, warning episodes, episode durations, and false warnings.

Inputs:
  data/processed/expanded_flight_split.csv
  <AEGIS_DATA_DIR>/<flight_name>/
  models/expanded_scaler.joblib
  models/expanded_mlp.pt

Outputs:
  data/processed/causal_replay_evaluation_metrics.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.streaming_failure_predictor import StreamingFailurePredictor, FEATURE_COLS


def extract_failure_episodes(df_stream: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract contiguous failure episodes where actual is_failure == 1."""
    episodes = []
    in_ep = False
    start_idx = None

    for i, row in df_stream.iterrows():
        is_f = row["actual_is_failure"]
        if is_f == 1:
            if not in_ep:
                in_ep = True
                start_idx = i
        else:
            if in_ep:
                in_ep = False
                end_idx = i - 1
                t_start = df_stream.loc[start_idx, "timestamp"]
                t_end = df_stream.loc[end_idx, "timestamp"]
                episodes.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": t_start,
                    "end_time": t_end,
                    "frame_count": end_idx - start_idx + 1,
                    "duration_s": t_end - t_start,
                })
    if in_ep:
        end_idx = len(df_stream) - 1
        t_start = df_stream.loc[start_idx, "timestamp"]
        t_end = df_stream.loc[end_idx, "timestamp"]
        episodes.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": t_start,
            "end_time": t_end,
            "frame_count": end_idx - start_idx + 1,
            "duration_s": t_end - t_start,
        })
    return episodes


def extract_warning_episodes(df_stream: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract contiguous warning episodes where predicted_label == 1."""
    episodes = []
    in_ep = False
    start_idx = None

    for i, row in df_stream.iterrows():
        if not row["has_prediction"]:
            continue
        p = row["pred_label"]
        if p == 1:
            if not in_ep:
                in_ep = True
                start_idx = i
        else:
            if in_ep:
                in_ep = False
                end_idx = i - 1
                t_start = df_stream.loc[start_idx, "timestamp"]
                t_end = df_stream.loc[end_idx, "timestamp"]
                episodes.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_time": t_start,
                    "end_time": t_end,
                    "frame_count": end_idx - start_idx + 1,
                    "duration_s": t_end - t_start,
                })
    if in_ep:
        end_idx = len(df_stream) - 1
        t_start = df_stream.loc[start_idx, "timestamp"]
        t_end = df_stream.loc[end_idx, "timestamp"]
        episodes.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": t_start,
            "end_time": t_end,
            "frame_count": end_idx - start_idx + 1,
            "duration_s": t_end - t_start,
        })
    return episodes


def replay_flight(
    flight_id: str,
    data_dir: Path,
    predictor: StreamingFailurePredictor,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Replay a single flight strictly in timestamp order through StreamingFailurePredictor."""
    gt_path = data_dir / flight_id / "dataset_gt.csv"
    raw_path = data_dir / flight_id / "raw_vo.csv"

    gt = pd.read_csv(gt_path)
    raw = pd.read_csv(raw_path)

    # Active window filter
    active_idx = gt["pos_z"].astype(float) >= 2.0
    t0 = gt.loc[active_idx, "timestamp_total_sec"].astype(float).min()
    t1 = gt.loc[active_idx, "timestamp_total_sec"].astype(float).max()

    raw_t = raw["timestamp_total_sec"].astype(float)
    in_window = (raw_t >= t0) & (raw_t <= t1)
    df_active = raw.loc[in_window].copy().reset_index(drop=True)

    predictor.reset()
    records = []

    # Causal streaming loop
    for i, row in df_active.iterrows():
        # ONLY current telemetry provided: NO VO pose, NO future data
        telemetry = {
            "timestamp": float(row["timestamp_total_sec"]),
            "eis_yaw_rate_deg": float(row["eis_yaw_rate_deg"]),
            "feature_vel_mean": float(row["feature_vel_mean"]),
            "is_r_frame": int(row["is_r_frame"]),
        }
        res = predictor.update(telemetry)

        # Ground truth attached ONLY for post-hoc validation
        inliers = int(row["num_inliers_pose"])
        is_failure = 1 if inliers < 8 else 0

        rec = {
            "flight_id": flight_id,
            "frame_idx": i,
            "timestamp": telemetry["timestamp"],
            "has_prediction": res["has_prediction"],
            "predicted_prob": res["prob"],
            "pred_label": res["pred_label"],
            "feat_lat_us": res["latencies_us"]["feature_construction_us"],
            "scale_lat_us": res["latencies_us"]["scaler_transform_us"],
            "mlp_lat_us": res["latencies_us"]["mlp_inference_us"],
            "e2e_lat_us": res["latencies_us"]["end_to_end_us"],
            "actual_inliers": inliers,
            "actual_is_failure": is_failure,
        }
        records.append(rec)

    df_out = pd.DataFrame(records)

    # Post-flight evaluation: compute K=5 forward window label
    forward_shifts = pd.concat([df_out["actual_is_failure"].shift(-step) for step in range(6)], axis=1)
    df_out["k5_label"] = forward_shifts.max(axis=1)

    # Drop warm-up (first 5 frames) and boundary end (last 5 frames)
    valid_mask = df_out["has_prediction"] & df_out["k5_label"].notna()
    df_eval = df_out[valid_mask].copy().reset_index(drop=True)

    return df_out, df_eval


import argparse
import os


def replay_flight_from_frames(
    flight_id: str,
    df_flight: pd.DataFrame,
    predictor: StreamingFailurePredictor,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute causal streaming replay from telemetry frames DataFrame."""
    df_flight = df_flight.sort_values("timestamp").reset_index(drop=True)
    predictor.reset()
    records = []

    for i, row in df_flight.iterrows():
        telemetry = {
            "timestamp": float(row["timestamp"]),
            "eis_yaw_rate_deg": float(row["eis_yaw_rate_deg"]),
            "feature_vel_mean": float(row["feature_vel_mean"]),
            "is_r_frame": int(row["is_r_frame"]),
        }
        res = predictor.update(telemetry)

        inliers = int(row["num_inliers_pose"])
        is_failure = 1 if inliers < 8 else 0

        rec = {
            "flight_id": flight_id,
            "frame_idx": i,
            "timestamp": telemetry["timestamp"],
            "has_prediction": res["has_prediction"],
            "predicted_prob": res["prob"],
            "pred_label": res["pred_label"],
            "feat_lat_us": res["latencies_us"]["feature_construction_us"],
            "scale_lat_us": res["latencies_us"]["scaler_transform_us"],
            "mlp_lat_us": res["latencies_us"]["mlp_inference_us"],
            "e2e_lat_us": res["latencies_us"]["end_to_end_us"],
            "actual_inliers": inliers,
            "actual_is_failure": is_failure,
        }
        records.append(rec)

    df_out = pd.DataFrame(records)
    forward_shifts = pd.concat([df_out["actual_is_failure"].shift(-step) for step in range(6)], axis=1)
    df_out["k5_label"] = forward_shifts.max(axis=1)

    valid_mask = df_out["has_prediction"] & df_out["k5_label"].notna()
    df_eval = df_out[valid_mask].copy().reset_index(drop=True)

    return df_out, df_eval


def main():
    parser = argparse.ArgumentParser(description="Causal Replay Evaluation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("AEGIS_DATA_DIR", Path(__file__).resolve().parents[1] / "data" / "raw")),
        help="Path to raw ROS dataset directory",
    )
    parser.add_argument(
        "--frames-file",
        type=Path,
        default=None,
        help="Path to telemetry_frames.csv.gz to replay directly from frames",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path to save evaluation metrics JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "models"
    data_dir = args.data_dir

    print("=================================================================")
    print("Research 2 — Causal Replay & Latency / Lead-Time Evaluation")
    print("=================================================================")

    split_csv = processed_dir / "expanded_flight_split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing {split_csv}")

    df_split = pd.read_csv(split_csv)
    test_flights = sorted(df_split[df_split["split"] == "test"]["run_dir"].tolist())
    print(f"Loaded {len(test_flights)} held-out test flights for causal replay.")

    scaler_path = models_dir / "expanded_scaler.joblib"
    model_path = models_dir / "expanded_mlp.pt"
    predictor = StreamingFailurePredictor(scaler_path, model_path, threshold=0.5)

    all_dfs = []
    all_evals = []
    flight_summaries = []

    # Latency accumulators
    all_feat_lats = []
    all_scale_lats = []
    all_mlp_lats = []
    all_e2e_lats = []

    # Failure episode counters
    total_episodes = 0
    true_early_warnings = 0
    post_failure_detections = 0
    missed_failures = 0
    lead_times_s = []

    # Warning episode counters
    total_warning_episodes = 0
    warning_durations_s = []
    positive_transitions = 0
    false_warning_episodes = 0

    telemetry_frames_map = None
    if args.frames_file is not None and args.frames_file.exists():
        print(f"Using pre-extracted frames from {args.frames_file}...")
        df_frames_all = pd.read_csv(args.frames_file)
        telemetry_frames_map = {f: grp for f, grp in df_frames_all.groupby("run_dir")}

    print("\n--- Executing Causal Streaming Replay Across Test Flights ---")
    for f_idx, fl in enumerate(test_flights, 1):
        if telemetry_frames_map is not None:
            if fl not in telemetry_frames_map:
                raise ValueError(f"Flight {fl} not found in {args.frames_file}")
            df_stream, df_eval = replay_flight_from_frames(fl, telemetry_frames_map[fl], predictor)
        else:
            df_stream, df_eval = replay_flight(fl, data_dir, predictor)
        all_dfs.append(df_stream)
        all_evals.append(df_eval)

        # Collect valid latencies
        valid_stream = df_stream[df_stream["has_prediction"]]
        all_feat_lats.extend(valid_stream["feat_lat_us"].tolist())
        all_scale_lats.extend(valid_stream["scale_lat_us"].tolist())
        all_mlp_lats.extend(valid_stream["mlp_lat_us"].tolist())
        all_e2e_lats.extend(valid_stream["e2e_lat_us"].tolist())

        # Extract failure episodes
        fl_episodes = extract_failure_episodes(df_stream)
        fl_warn_episodes = extract_warning_episodes(df_stream)

        # Count positive transitions (0 -> 1)
        preds = valid_stream["pred_label"].values
        trans = int((np.diff(preds) == 1).sum()) if len(preds) > 1 else 0
        positive_transitions += trans

        fl_true_warn = 0
        fl_post_det = 0
        fl_missed = 0
        fl_leads = []

        for ep in fl_episodes:
            total_episodes += 1
            s_i = ep["start_idx"]
            e_i = ep["end_idx"]
            t_onset = ep["start_time"]

            # Pre-failure window: frames preceding onset within K=5 (~0.165s)
            pre_win = df_stream.loc[max(0, s_i - 5): s_i - 1]
            pre_warns = pre_win[pre_win["pred_label"] == 1]

            # In-failure window
            during_win = df_stream.loc[s_i: e_i]
            during_warns = during_win[during_win["pred_label"] == 1]

            if len(pre_warns) > 0:
                true_early_warnings += 1
                fl_true_warn += 1
                # Earliest warning timestamp in the pre-failure sequence
                t_first_warn = pre_warns["timestamp"].min()
                lead = t_onset - t_first_warn
                lead_times_s.append(lead)
                fl_leads.append(lead)
            elif len(during_warns) > 0:
                post_failure_detections += 1
                fl_post_det += 1
            else:
                missed_failures += 1
                fl_missed += 1

        # Check false warning episodes: warnings that do NOT overlap with any failure within K=5
        for w_ep in fl_warn_episodes:
            total_warning_episodes += 1
            warning_durations_s.append(w_ep["duration_s"])
            # Check if any failure occurs between w_ep["start_time"] and w_ep["end_time"] + 0.165
            w_start = w_ep["start_idx"]
            w_end = min(len(df_stream) - 1, w_ep["end_idx"] + 5)
            has_fail = (df_stream.loc[w_start:w_end, "actual_is_failure"] == 1).any()
            if not has_fail:
                false_warning_episodes += 1

        # Flight-level metrics
        y_true = df_eval["k5_label"].values.astype(int)
        y_prob = df_eval["predicted_prob"].values
        y_pred = df_eval["pred_label"].values

        has_both = len(np.unique(y_true)) > 1
        fl_auc = float(roc_auc_score(y_true, y_prob)) if has_both else float("nan")
        fl_f1 = float(f1_score(y_true, y_pred, zero_division=0))

        flight_summaries.append({
            "flight": fl,
            "active_frames": len(df_stream),
            "eval_frames": len(df_eval),
            "episodes": len(fl_episodes),
            "early_warns": fl_true_warn,
            "post_dets": fl_post_det,
            "missed": fl_missed,
            "mean_lead_s": float(np.mean(fl_leads)) if fl_leads else 0.0,
            "auroc": fl_auc,
            "f1": fl_f1,
        })
        print(f"  [{f_idx:2d}/14] {fl:14s} | Frames: {len(df_stream):4d} | Ep: {len(fl_episodes):2d} "
              f"(Early: {fl_true_warn:2d}, Post: {fl_post_det:2d}, Missed: {fl_missed:2d}) | "
              f"AUROC: {fl_auc:.4f} | F1: {fl_f1:.4f}")

    # Aggregate Evaluation
    df_total_eval = pd.concat(all_evals, ignore_index=True)
    y_all_true = df_total_eval["k5_label"].values.astype(int)
    y_all_prob = df_total_eval["predicted_prob"].values
    y_all_pred = df_total_eval["pred_label"].values

    cm = confusion_matrix(y_all_true, y_all_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    overall_metrics = {
        "auroc": float(roc_auc_score(y_all_true, y_all_prob)),
        "auprc": float(average_precision_score(y_all_true, y_all_prob)),
        "f1": float(f1_score(y_all_true, y_all_pred, zero_division=0)),
        "precision": float(precision_score(y_all_true, y_all_pred, zero_division=0)),
        "recall": float(recall_score(y_all_true, y_all_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_all_true, y_all_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": len(df_total_eval),
        "positive_rate": float(y_all_true.mean() * 100),
    }

    # Latency statistics
    def calc_stats(arr):
        a = np.array(arr)
        return {
            "mean_us": float(np.mean(a)),
            "median_us": float(np.median(a)),
            "p95_us": float(np.percentile(a, 95)),
            "p99_us": float(np.percentile(a, 99)),
            "max_us": float(np.max(a)),
            "mean_ms": float(np.mean(a) / 1000.0),
            "median_ms": float(np.median(a) / 1000.0),
            "p95_ms": float(np.percentile(a, 95) / 1000.0),
            "p99_ms": float(np.percentile(a, 99) / 1000.0),
            "max_ms": float(np.max(a) / 1000.0),
        }

    latency_report = {
        "feature_construction": calc_stats(all_feat_lats),
        "scaler_transform": calc_stats(all_scale_lats),
        "mlp_inference": calc_stats(all_mlp_lats),
        "end_to_end": calc_stats(all_e2e_lats),
        "telemetry_sample_period_ms": 33.333,
        "telemetry_hz": 30.0,
        "headroom_pct": float((33.333 - (np.percentile(all_e2e_lats, 99) / 1000.0)) / 33.333 * 100.0),
    }

    # Lead time statistics
    lead_arr = np.array(lead_times_s)
    lead_report = {
        "total_failure_episodes": total_episodes,
        "true_early_warnings": true_early_warnings,
        "early_warning_rate_pct": float(true_early_warnings / total_episodes * 100.0) if total_episodes > 0 else 0.0,
        "post_failure_detections": post_failure_detections,
        "post_failure_detection_rate_pct": float(post_failure_detections / total_episodes * 100.0) if total_episodes > 0 else 0.0,
        "missed_failures": missed_failures,
        "missed_failure_rate_pct": float(missed_failures / total_episodes * 100.0) if total_episodes > 0 else 0.0,
        "lead_time_stats_s": {
            "mean_s": float(np.mean(lead_arr)) if len(lead_arr) > 0 else 0.0,
            "median_s": float(np.median(lead_arr)) if len(lead_arr) > 0 else 0.0,
            "p25_s": float(np.percentile(lead_arr, 25)) if len(lead_arr) > 0 else 0.0,
            "p75_s": float(np.percentile(lead_arr, 75)) if len(lead_arr) > 0 else 0.0,
            "min_s": float(np.min(lead_arr)) if len(lead_arr) > 0 else 0.0,
            "max_s": float(np.max(lead_arr)) if len(lead_arr) > 0 else 0.0,
        },
    }

    # Warning stability report
    dur_arr = np.array(warning_durations_s)
    stability_report = {
        "total_warning_episodes": total_warning_episodes,
        "positive_transitions": positive_transitions,
        "false_warning_episodes": false_warning_episodes,
        "false_warning_rate_pct": float(false_warning_episodes / total_warning_episodes * 100.0) if total_warning_episodes > 0 else 0.0,
        "mean_warning_duration_s": float(np.mean(dur_arr)) if len(dur_arr) > 0 else 0.0,
        "median_warning_duration_s": float(np.median(dur_arr)) if len(dur_arr) > 0 else 0.0,
        "max_warning_duration_s": float(np.max(dur_arr)) if len(dur_arr) > 0 else 0.0,
    }

    payload = {
        "overall_metrics": overall_metrics,
        "latency_report": latency_report,
        "lead_report": lead_report,
        "stability_report": stability_report,
        "flight_summaries": flight_summaries,
    }

    if args.out_json is not None:
        out_json = args.out_json
    elif args.frames_file is not None:
        out_json = repo_root / "results" / "audit" / "causal_replay_from_frames_metrics.json"
    else:
        out_json = processed_dir / "causal_replay_evaluation_metrics.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics JSON to: {out_json.name}")

    # Print summary
    print("\n=================================================================")
    print("CAUSAL REPLAY SUMMARY ACROSS 14 HELD-OUT TEST FLIGHTS")
    print("=================================================================")
    print(f"Overall Metrics (N={overall_metrics['total_samples']:,} samples):")
    print(f"  AUROC    : {overall_metrics['auroc']:.4f}")
    print(f"  AUPRC    : {overall_metrics['auprc']:.4f}")
    print(f"  F1       : {overall_metrics['f1']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall   : {overall_metrics['recall']:.4f}")
    print(f"  CM       : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\nComputational Latency:")
    print(f"  Feature Construction: Mean={latency_report['feature_construction']['mean_us']:.1f}us, P99={latency_report['feature_construction']['p99_us']:.1f}us")
    print(f"  Scaler Transform    : Mean={latency_report['scaler_transform']['mean_us']:.1f}us, P99={latency_report['scaler_transform']['p99_us']:.1f}us")
    print(f"  MLP Inference       : Mean={latency_report['mlp_inference']['mean_us']:.1f}us, P99={latency_report['mlp_inference']['p99_us']:.1f}us")
    print(f"  End-to-End Latency  : Mean={latency_report['end_to_end']['mean_ms']:.4f}ms, Median={latency_report['end_to_end']['median_ms']:.4f}ms, P95={latency_report['end_to_end']['p95_ms']:.4f}ms, P99={latency_report['end_to_end']['p99_ms']:.4f}ms, Max={latency_report['end_to_end']['max_ms']:.4f}ms")
    print(f"  Telemetry Period    : 33.333 ms (~30 Hz)")
    print(f"  Available Headroom  : {latency_report['headroom_pct']:.2f}% (P99 uses only {(100 - latency_report['headroom_pct']):.2f}% of budget)")

    print("\nEarly Warning vs. Detection:")
    print(f"  Total Failure Episodes  : {lead_report['total_failure_episodes']}")
    print(f"  True Early Warnings     : {lead_report['true_early_warnings']} ({lead_report['early_warning_rate_pct']:.1f}%)")
    print(f"  Post-Failure Detections : {lead_report['post_failure_detections']} ({lead_report['post_failure_detection_rate_pct']:.1f}%)")
    print(f"  Missed Failures         : {lead_report['missed_failures']} ({lead_report['missed_failure_rate_pct']:.1f}%)")
    print(f"  Lead Time (s)           : Median={lead_report['lead_time_stats_s']['median_s']:.3f}s, Mean={lead_report['lead_time_stats_s']['mean_s']:.3f}s, P25={lead_report['lead_time_stats_s']['p25_s']:.3f}s, P75={lead_report['lead_time_stats_s']['p75_s']:.3f}s, Min={lead_report['lead_time_stats_s']['min_s']:.3f}s, Max={lead_report['lead_time_stats_s']['max_s']:.3f}s")


if __name__ == "__main__":
    main()
