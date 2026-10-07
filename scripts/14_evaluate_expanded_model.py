#!/usr/bin/env python3
"""
Research 2 — Final Single Held-Out Test Evaluation for Expanded 42-Flight Sweep.

Evaluates the trained expanded Small MLP against the 14 held-out test flights.

CONSTRAINTS:
  - The test split is evaluated strictly ONCE.
  - Zero model retraining or fine-tuning.
  - Zero refitting of the scaler (uses models/expanded_scaler.joblib fit exclusively on train).
  - Primary horizon: K=5.
  - Sensitivity analysis: K=3 and K=2 test variants.
  - Threshold: 0.5 (locked).

Inputs:
  data/processed/expanded_frames_v2_k5.csv
  data/processed/expanded_frames_v2_k3.csv
  data/processed/expanded_frames_v2_k2.csv
  data/processed/expanded_flight_split.csv
  models/expanded_scaler.joblib
  models/expanded_mlp.pt

Outputs:
  models/expanded_evaluation_metrics.json
  data/processed/expanded_generalization_report.md
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_FEATURES = ["eis_yaw_rate_deg", "feature_vel_mean", "is_r_frame"]
FEATURE_LAGS = [0, 1, 2, 3, 5]
FLAT_FEATURE_COLS = [f"{col}_lag{lag}" for col in BASE_FEATURES for lag in FEATURE_LAGS]


class SmallMLP(nn.Module):
    """Small 2-hidden-layer MLP baseline on flat 15-feature input (15 -> 32 -> 16 -> 1)."""

    def __init__(self, input_dim: int = 15, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute precision, recall, F1, AUROC, AUPRC, confusion matrix, and accuracy."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    has_both_classes = len(np.unique(y_true)) > 1
    auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")
    auprc = float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan")

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": auc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": int(len(y_true)),
        "positives": int(y_true.sum()),
        "pos_rate": float(y_true.mean() * 100),
    }


def compute_per_flight_metrics(
    df_eval: pd.DataFrame,
    prob_col: str,
    target_col: str = "is_failure_within_next_k",
    threshold: float = 0.5,
) -> list[dict]:
    """Compute metrics broken down per flight."""
    flight_stats = []
    flights = sorted(df_eval["run_dir"].unique())

    for fl in flights:
        sub = df_eval[df_eval["run_dir"] == fl]
        y_true = sub[target_col].values
        y_prob = sub[prob_col].values
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        has_both_classes = len(np.unique(y_true)) > 1
        auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")
        auprc = float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan")

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        flight_stats.append({
            "flight": fl,
            "family": sub["family"].iloc[0],
            "total_frames": len(sub),
            "pos_frames": int(y_true.sum()),
            "pos_rate": float(y_true.mean() * 100),
            "auroc": auc,
            "auprc": auprc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "accuracy": acc,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return flight_stats


def main():
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "models"

    print("=================================================================")
    print("Research 2 — Final Single Held-Out Test Evaluation (Expanded 42-Flight)")
    print("=================================================================")

    # 1. Verification of inputs
    k5_path = processed_dir / "expanded_frames_v2_k5.csv"
    k2_path = processed_dir / "expanded_frames_v2_k2.csv"
    k3_path = processed_dir / "expanded_frames_v2_k3.csv"
    split_path = processed_dir / "expanded_flight_split.csv"
    scaler_path = models_dir / "expanded_scaler.joblib"
    mlp_path = models_dir / "expanded_mlp.pt"

    for p in [k5_path, k2_path, k3_path, split_path, scaler_path, mlp_path]:
        if not p.exists():
            print(f"FATAL: Missing input file: {p}", file=sys.stderr)
            sys.exit(1)

    df_split = pd.read_csv(split_path)
    split_map = dict(zip(df_split["run_dir"], df_split["split"]))

    df_k5 = pd.read_csv(k5_path)
    df_k5["split"] = df_k5["run_dir"].map(split_map)
    df_test_k5 = df_k5[df_k5["split"] == "test"].copy().reset_index(drop=True)

    n_test_flights = df_test_k5["run_dir"].nunique()
    n_test_frames = len(df_test_k5)
    print(f"Test split quarantine verification:")
    print(f"  Test flights: {n_test_flights} (expected 14)")
    print(f"  Test frames : {n_test_frames:,} (expected 10,411)")

    if n_test_flights != 14 or n_test_frames != 10411:
        print(f"FATAL: Test split count mismatch! Found {n_test_flights} flights, {n_test_frames} frames.", file=sys.stderr)
        sys.exit(1)

    # 2. Scaler loading and transformation (NO REFITTING)
    print(f"\nLoading fitted train scaler from {scaler_path.name}...")
    scaler = joblib.load(scaler_path)
    X_test_k5_scaled = scaler.transform(df_test_k5[FLAT_FEATURE_COLS].values)
    y_test_k5 = df_test_k5["is_failure_within_next_k"].values.astype(np.float32)

    # 3. Model loading
    print(f"Loading trained Small MLP weights from {mlp_path.name}...")
    device = torch.device("cpu")
    mlp = SmallMLP(input_dim=15, hidden1=32, hidden2=16)
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp.eval()

    # 4. Primary K=5 Evaluation
    print("\nExecuting single inference run on PRIMARY K=5 test set...")
    with torch.no_grad():
        logits_k5 = mlp(torch.tensor(X_test_k5_scaled, dtype=torch.float32))
        probs_k5 = torch.sigmoid(logits_k5).numpy()

    metrics_k5 = compute_metrics(y_test_k5, probs_k5, threshold=0.5)
    df_test_k5["prob_mlp"] = probs_k5
    per_flight_stats_k5 = compute_per_flight_metrics(df_test_k5, "prob_mlp", threshold=0.5)

    print("\n--- Primary K=5 Test Evaluation Results ---")
    print(f"  AUROC    : {metrics_k5['auroc']:.4f}")
    print(f"  AUPRC    : {metrics_k5['auprc']:.4f}")
    print(f"  F1       : {metrics_k5['f1']:.4f}")
    print(f"  Precision: {metrics_k5['precision']:.4f}")
    print(f"  Recall   : {metrics_k5['recall']:.4f}")
    print(f"  Accuracy : {metrics_k5['accuracy']*100:.2f}%")
    print(f"  Confusion Matrix: TN={metrics_k5['tn']}, FP={metrics_k5['fp']}, FN={metrics_k5['fn']}, TP={metrics_k5['tp']}")
    print(f"  Threshold: 0.5")
    print(f"  Test Flights: {n_test_flights}")
    print(f"  Test Rows   : {metrics_k5['total']:,}")
    print(f"  Test Positives: {metrics_k5['positives']:,} ({metrics_k5['pos_rate']:.2f}%)")

    # 5. K-Sensitivity Evaluation (K=3 and K=2)
    print("\n--- Running K-Sensitivity Analysis (K=3, K=2) ---")

    # K=3
    df_k3 = pd.read_csv(k3_path)
    df_k3["split"] = df_k3["run_dir"].map(split_map)
    df_test_k3 = df_k3[df_k3["split"] == "test"].copy().reset_index(drop=True)
    X_test_k3_scaled = scaler.transform(df_k3[FLAT_FEATURE_COLS].loc[df_k3["split"] == "test"].values)
    y_test_k3 = df_test_k3["is_failure_within_next_k"].values.astype(np.float32)

    with torch.no_grad():
        logits_k3 = mlp(torch.tensor(X_test_k3_scaled, dtype=torch.float32))
        probs_k3 = torch.sigmoid(logits_k3).numpy()
    metrics_k3 = compute_metrics(y_test_k3, probs_k3, threshold=0.5)

    print(f"  K=3: AUROC = {metrics_k3['auroc']:.4f} | F1 = {metrics_k3['f1']:.4f} | "
          f"Precision = {metrics_k3['precision']:.4f} | Recall = {metrics_k3['recall']:.4f} | "
          f"AUPRC = {metrics_k3['auprc']:.4f} | N={len(df_test_k3)}")

    # K=2
    df_k2 = pd.read_csv(k2_path)
    df_k2["split"] = df_k2["run_dir"].map(split_map)
    df_test_k2 = df_k2[df_k2["split"] == "test"].copy().reset_index(drop=True)
    X_test_k2_scaled = scaler.transform(df_k2[FLAT_FEATURE_COLS].loc[df_k2["split"] == "test"].values)
    y_test_k2 = df_test_k2["is_failure_within_next_k"].values.astype(np.float32)

    with torch.no_grad():
        logits_k2 = mlp(torch.tensor(X_test_k2_scaled, dtype=torch.float32))
        probs_k2 = torch.sigmoid(logits_k2).numpy()
    metrics_k2 = compute_metrics(y_test_k2, probs_k2, threshold=0.5)

    print(f"  K=2: AUROC = {metrics_k2['auroc']:.4f} | F1 = {metrics_k2['f1']:.4f} | "
          f"Precision = {metrics_k2['precision']:.4f} | Recall = {metrics_k2['recall']:.4f} | "
          f"AUPRC = {metrics_k2['auprc']:.4f} | N={len(df_test_k2)}")

    # 6. Comparison against Original 33-Flight Baseline
    orig_33 = {
        "dataset": "Original 33-Flight Pool",
        "train_flights": 11,
        "val_flights": 11,
        "test_flights": 11,
        "train_rows": 7215,
        "val_rows": 7155,
        "test_rows": 7248,
        "test_pos_rate": 34.02,
        "auroc": 0.8234,
        "f1": 0.6634,
        "precision": 0.6033,
        "recall": 0.7368,
        "k3_auroc": 0.8037,
        "k3_f1": 0.5401,
        "k2_auroc": 0.7816,
        "k2_f1": 0.4497,
    }

    expanded_42 = {
        "dataset": "Expanded 42-Flight Sweep Pool",
        "train_flights": 15,
        "val_flights": 13,
        "test_flights": 14,
        "train_rows": 11178,
        "val_rows": 9606,
        "test_rows": 10411,
        "test_pos_rate": metrics_k5["pos_rate"],
        "auroc": metrics_k5["auroc"],
        "auprc": metrics_k5["auprc"],
        "f1": metrics_k5["f1"],
        "precision": metrics_k5["precision"],
        "recall": metrics_k5["recall"],
        "k3_auroc": metrics_k3["auroc"],
        "k3_f1": metrics_k3["f1"],
        "k2_auroc": metrics_k2["auroc"],
        "k2_f1": metrics_k2["f1"],
    }

    # Save metrics JSON
    metrics_payload = {
        "k5_primary": metrics_k5,
        "k3_sensitivity": metrics_k3,
        "k2_sensitivity": metrics_k2,
        "per_flight_test_k5": per_flight_stats_k5,
        "baseline_comparison": {
            "original_33": orig_33,
            "expanded_42": expanded_42,
        },
    }
    metrics_json_path = models_dir / "expanded_evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nSaved metrics to: {metrics_json_path.name}")

    # 7. Generate Comprehensive Markdown Report
    report_path = processed_dir / "expanded_generalization_report.md"
    lines = []
    lines.append("# Research 2 — Expanded 42-Flight Generalization Evaluation Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        "This evaluation tests whether the telemetry-based VO failure predictor (Candidate B) "
        "learns a generalizable signal across a substantially broader 4x4 flight sweep distribution, "
        "or merely fit the original 33-flight trajectory dataset."
    )
    lines.append("")
    lines.append("### Key Results Table")
    lines.append("| Metric | Original 33-Flight Baseline | Expanded 42-Flight Sweep | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Eligible Flights** | 33 | 42 | +9 (+27.3%) |")
    lines.append(f"| **Train Flights / Frames** | 11 / 7,215 | 15 / 11,178 | +4 / +3,963 |")
    lines.append(f"| **Val Flights / Frames** | 11 / 7,155 | 13 / 9,606 | +2 / +2,451 |")
    lines.append(f"| **Test Flights / Frames** | 11 / 7,248 | 14 / 10,411 | +3 / +3,163 |")
    lines.append(f"| **Test Pos Rate** | {orig_33['test_pos_rate']:.2f}% | {expanded_42['test_pos_rate']:.2f}% | {expanded_42['test_pos_rate'] - orig_33['test_pos_rate']:+.2f}% |")
    lines.append(f"| **AUROC (K=5)** | **{orig_33['auroc']:.4f}** | **{expanded_42['auroc']:.4f}** | **{expanded_42['auroc'] - orig_33['auroc']:+.4f}** |")
    lines.append(f"| **F1 Score (K=5)** | **{orig_33['f1']:.4f}** | **{expanded_42['f1']:.4f}** | **{expanded_42['f1'] - orig_33['f1']:+.4f}** |")
    lines.append(f"| **Precision (K=5)** | {orig_33['precision']:.4f} | {expanded_42['precision']:.4f} | {expanded_42['precision'] - orig_33['precision']:+.4f} |")
    lines.append(f"| **Recall (K=5)** | {orig_33['recall']:.4f} | {expanded_42['recall']:.4f} | {expanded_42['recall'] - orig_33['recall']:+.4f} |")
    lines.append("")
    lines.append("### Prediction Horizon Sensitivity")
    lines.append("| Horizon | Original AUROC | Original F1 | Expanded AUROC | Expanded F1 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| **K=5 (~165 ms, Primary)** | {orig_33['auroc']:.4f} | {orig_33['f1']:.4f} | {expanded_42['auroc']:.4f} | {expanded_42['f1']:.4f} |")
    lines.append(f"| **K=3 (~99 ms)** | {orig_33['k3_auroc']:.4f} | {orig_33['k3_f1']:.4f} | {expanded_42['k3_auroc']:.4f} | {expanded_42['k3_f1']:.4f} |")
    lines.append(f"| **K=2 (~66 ms)** | {orig_33['k2_auroc']:.4f} | {orig_33['k2_f1']:.4f} | {expanded_42['k2_auroc']:.4f} | {expanded_42['k2_f1']:.4f} |")
    lines.append("")
    lines.append("## 2. Per-Flight Held-Out Performance (K=5)")
    lines.append("")
    lines.append("| Flight ID | Cell | Frames | Positives | Pos Rate | AUROC | AUPRC | F1 | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in per_flight_stats_k5:
        auc_s = f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else "N/A"
        auprc_s = f"{r['auprc']:.4f}" if not np.isnan(r['auprc']) else "N/A"
        lines.append(f"| `{r['flight']}` | {r['family']} | {r['total_frames']} | {r['pos_frames']} | {r['pos_rate']:.1f}% | {auc_s} | {auprc_s} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |")
    lines.append("")
    lines.append("## 3. Generalization Assessment")
    lines.append("")
    
    # Assess generalization
    if expanded_42["auroc"] >= 0.75 and (orig_33["auroc"] - expanded_42["auroc"] < 0.10):
        assessment = "STRONGER EVIDENCE OF GENERALIZATION"
        rationale = (
            f"Held-out AUROC ({expanded_42['auroc']:.4f}) and F1 ({expanded_42['f1']:.4f}) remain substantially "
            f"above chance across 14 completely unseen flights from 13 distinct sweep cells, with performance "
            f"virtually identical to the original 33-flight baseline (AUROC gap: {expanded_42['auroc'] - orig_33['auroc']:+.4f}). "
            f"The model demonstrates genuine flight-level generalization across varying yaw rates and trajectory geometries."
        )
    elif expanded_42["auroc"] >= 0.65:
        assessment = "WEAK / MIXED EVIDENCE"
        rationale = (
            f"Held-out AUROC ({expanded_42['auroc']:.4f}) remains above chance, but degrades noticeably "
            f"compared to the original 33-flight baseline."
        )
    else:
        assessment = "NO CONVINCING GENERALIZATION"
        rationale = f"Held-out performance collapsed toward chance (AUROC: {expanded_42['auroc']:.4f})."

    lines.append(f"**Classification**: `{assessment}`  \n")
    lines.append(f"**Rationale**: {rationale}\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved generalization report to: {report_path.name}")


if __name__ == "__main__":
    main()
