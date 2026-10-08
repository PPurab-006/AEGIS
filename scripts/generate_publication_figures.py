#!/usr/bin/env python3
"""
Research 2 — Publication-Quality Figure Set Generator.

Generates the complete 5-figure publication set for the Research 2 preprint:
  FIGURE 1: Research pipeline / experimental design (schematic)
  FIGURE 2: Main held-out predictive performance (ROC, PR, Threshold characterization)
  FIGURE 3: Temporal early-warning behavior (Telemetry, Probability, VO Inliers, Lead time)
  FIGURE 4: Cross-flight generalization across 14 held-out test flights
  FIGURE 5: Real-time feasibility and latency distribution in ROS 2 middleware

All outputs are saved as vector PDF, SVG, and 300 DPI PNG in:
  results/figures/publication/

This script does NOT modify any models, data, thresholds, or splits.
It uses strictly the frozen experimental outputs as source of truth.
"""

import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# GLOBAL PUBLICATION STYLING CONFIGURATION
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "publication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Curated publication color palette
NAVY = "#1a5276"
BLUE = "#2980b9"
TEAL = "#16a085"
GREEN = "#27ae60"
AMBER = "#d35400"
ORANGE = "#e67e22"
RED = "#c0392b"
PURPLE = "#8e44ad"
DARK_SLATE = "#2c3e50"
BORDER_GRAY = "#bdc3c7"


# SmallMLP model class matching repository definition
class SmallMLP(nn.Module):
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


# ---------------------------------------------------------------------------
# FIGURE 1: RESEARCH PIPELINE & EXPERIMENTAL DESIGN SCHEMATIC
# ---------------------------------------------------------------------------
def generate_figure_1():
    """Figure 1: Clean, publication-quality experimental design and research pipeline."""
    print("\n[Fig 1] Generating research pipeline schematic...")

    fig = plt.figure(figsize=(7.2, 5.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)  # Top-down coordinates for exact vertical layout: 0 is top, 100 is bottom

    # Helper function for drawing rounded boxes
    def draw_box(x, y, w, h, title, subtitle="", bg="#ffffff", border="#2c3e50", lw=1.0, rx=1.2):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.2,rounding_size={rx}",
            facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.32, title, ha="center", va="center",
                    fontsize=8.0, fontweight="bold", color=border, zorder=3)
            ax.text(x + w / 2, y + h * 0.68, subtitle, ha="center", va="center",
                    fontsize=6.8, color="#333333", zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=7.8, fontweight="bold", color=border, zorder=3)

    # Helper for arrow connecting boxes
    def draw_arrow(x1, y1, x2, y2, label="", color="#555555", lw=1.1, label_pos="above"):
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=9,
            color=color, linewidth=lw, zorder=4
        )
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if label_pos == "above":
                ax.text(mx, my - 1.2, label, ha="center", va="bottom",
                        fontsize=6.7, color=color, fontweight="semibold", zorder=5)
            elif label_pos == "right":
                ax.text(mx + 1.2, my, label, ha="left", va="center",
                        fontsize=6.7, color=color, fontweight="semibold", zorder=5)
            elif label_pos == "below":
                ax.text(mx, my + 1.2, label, ha="center", va="top",
                        fontsize=6.7, color=color, fontweight="semibold", zorder=5)

    # =======================================================================
    # Container Row 1: Data Generation & Partitioning (y: 3 to 30)
    # =======================================================================
    row1_bg = patches.FancyBboxPatch(
        (1.5, 3.0), 97.0, 27.0, boxstyle="round,pad=0.4,rounding_size=1.5",
        facecolor="#f4f6f9", edgecolor="#d5dbdb", linewidth=0.8, zorder=1
    )
    ax.add_patch(row1_bg)
    ax.text(3.0, 4.8, "(a) Parametric Flight Sweep & Leak-Free Partitioning",
            fontsize=8.6, fontweight="bold", color=NAVY, va="top")

    # Step 1.1: 4x4 Flight Sweep
    draw_box(3.5, 7.5, 26.0, 20.5,
             "PX4/Gazebo Flight Sweep",
             "4 Yaw Rates (7.5°–90°/s)\n× 4 Geometries (C, B, S, H)\n48 Executions in Sim",
             bg="#ebf5fb", border=BLUE)

    draw_arrow(29.8, 17.75, 34.0, 17.75, "Physical\nGates", label_pos="above")

    # Step 1.2: Eligibility Filter
    draw_box(34.2, 7.5, 25.5, 20.5,
             "Quality Envelope Filter",
             "Excludes 6 Unviable Flights\n42 Eligible Flights\n31,195 Frames (34.6% Fail)",
             bg="#fef9e7", border=ORANGE)

    draw_arrow(60.0, 17.75, 64.2, 17.75, "Seed 42\nDisjoint", label_pos="above")

    # Step 1.3: Flight-Level Split
    draw_box(64.5, 7.5, 32.5, 20.5,
             "Flight-Level Quarantine",
             "Train: 15 flights (11,178 frames)\nVal: 13 flights (9,606 frames)\nTest: 14 flights (10,411 frames)\nTrain ∩ Test = ∅ (Zero Leak)",
             bg="#eafaf1", border=GREEN)

    # Connectors from Row 1 down to Row 2
    draw_arrow(16.5, 28.0, 16.5, 36.5, "Sensor Stream", color=BLUE, label_pos="right")
    draw_arrow(80.5, 28.0, 80.5, 36.5, "Scaler Fit (Train)", color=GREEN, label_pos="right")

    # =======================================================================
    # Container Row 2: Streaming Telemetry & Prediction Pipeline (y: 36.5 to 63.5)
    # =======================================================================
    row2_bg = patches.FancyBboxPatch(
        (1.5, 36.5), 97.0, 27.5, boxstyle="round,pad=0.4,rounding_size=1.5",
        facecolor="#fdfefe", edgecolor="#d5dbdb", linewidth=0.8, zorder=1
    )
    ax.add_patch(row2_bg)
    ax.text(3.0, 38.3, "(b) Causal Feature Construction & Lightweight Neural Predictor",
            fontsize=8.6, fontweight="bold", color=NAVY, va="top")

    # Step 2.1: 30 Hz Telemetry Ingestion
    draw_box(3.5, 41.0, 20.0, 21.0,
             "Flight Telemetry (30 Hz)",
             "• Yaw Rate: ω_z (°/s)\n• Optical Flow: v̄_flow (px)\n• Keyframe Flag: r ∈ {0,1}",
             bg="#ebf5fb", border=BLUE)

    draw_arrow(23.8, 51.5, 28.2, 51.5, "N=6 FIFO", label_pos="above")

    # Step 2.2: Rolling Buffer & Lagged Features
    draw_box(28.5, 41.0, 21.0, 21.0,
             "15-Dim Lagged Vector",
             "Lags {0, 1, 2, 3, 5}\nSpan = 165 ms\nx_t = [ω_lag, v̄_lag, r_lag]",
             bg="#f4ecf7", border=PURPLE)

    draw_arrow(49.8, 51.5, 54.2, 51.5, "Scale", label_pos="above")

    # Step 2.3: Frozen MLP & Prediction
    draw_box(54.5, 41.0, 22.0, 21.0,
             "SmallMLP Predictor",
             "15 → 32 → 16 → 1\n1,057 Parameters\nLatency: 0.128 ms CPU",
             bg="#fdf2e9", border=AMBER)

    draw_arrow(76.8, 51.5, 81.2, 51.5, "Sigmoid", label_pos="above")

    # Step 2.4: Failure Probability & Threshold Logic
    draw_box(81.5, 41.0, 15.5, 21.0,
             "Decision Logic",
             "P(fail_{t+5} | x_t)\nK=5 Forward Horizon\nθ = 0.50 Early Warning",
             bg="#fdedec", border=RED)

    # Connectors from Row 2 down to Row 3
    draw_arrow(18.0, 64.0, 18.0, 72.5, "Batch Offline", color=NAVY, label_pos="right")
    draw_arrow(50.0, 64.0, 50.0, 72.5, "Causal FIFO", color=PURPLE, label_pos="right")
    draw_arrow(82.0, 64.0, 82.0, 72.5, "Live ROS 2", color=GREEN, label_pos="right")

    # =======================================================================
    # Container Row 3: Three-Tier Empirical Verification (y: 72.5 to 98.0)
    # =======================================================================
    row3_bg = patches.FancyBboxPatch(
        (1.5, 72.5), 97.0, 25.5, boxstyle="round,pad=0.4,rounding_size=1.5",
        facecolor="#f4f6f9", edgecolor="#d5dbdb", linewidth=0.8, zorder=1
    )
    ax.add_patch(row3_bg)
    ax.text(3.0, 74.3, "(c) Three-Tier Empirical Verification Framework",
            fontsize=8.6, fontweight="bold", color=NAVY, va="top")

    # Tier 1: Offline Batch Evaluation
    draw_box(3.5, 77.0, 29.5, 19.0,
             "Tier 1: Offline Generalization",
             "14 Disjoint Held-Out Flights\nAUROC = 0.8046 | AUPRC = 0.7260\nF1 = 0.6484 (Prec 0.581, Rec 0.734)\nZero Optimization Leakage",
             bg="#ffffff", border=NAVY)

    # Tier 2: Causal Sequential Replay
    draw_box(35.2, 77.0, 29.5, 19.0,
             "Tier 2: Causal Sequential Replay",
             "10,481 Frames Replayed Sequentially\n79.3% True Early Warnings (522/658)\nMedian Lead Time: 0.165 s (5 Frames)\nAlgorithmic Latency: 0.128 ms",
             bg="#ffffff", border=PURPLE)

    # Tier 3: Live ROS 2 Deployment
    draw_box(67.0, 77.0, 30.0, 19.0,
             "Tier 3: Live ROS 2 Deployment",
             "Live ROS 2 Node (rclpy DDS)\n192 Hz Sustained Rate (>6× Headroom)\nMean Latency: 0.634 ms (P99 3.71 ms)\n93.4% Live Early Warning Rate",
             bg="#ffffff", border=GREEN)

    # Save outputs
    fig_name = "fig1_pipeline_schematic"
    plt.savefig(OUTPUT_DIR / f"{fig_name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.svg", format="svg", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {fig_name} (.pdf, .svg, .png)")


# ---------------------------------------------------------------------------
# FIGURE 2: MAIN HELD-OUT PREDICTIVE PERFORMANCE
# ---------------------------------------------------------------------------
def generate_figure_2():
    """Figure 2: Multi-panel held-out predictive performance (ROC, PR, Threshold sensitivity)."""
    print("\n[Fig 2] Generating held-out predictive performance figure...")

    # Load data and model
    df_split = pd.read_csv(REPO_ROOT / "data" / "processed" / "expanded_flight_split.csv")
    split_map = dict(zip(df_split["run_dir"], df_split["split"]))
    df_k5 = pd.read_csv(REPO_ROOT / "data" / "processed" / "expanded_frames_v2_k5.csv")
    df_k5["split"] = df_k5["run_dir"].map(split_map)
    df_test = df_k5[df_k5["split"] == "test"].copy().reset_index(drop=True)

    base_features = ["eis_yaw_rate_deg", "feature_vel_mean", "is_r_frame"]
    feature_lags = [0, 1, 2, 3, 5]
    flat_cols = [f"{col}_lag{lag}" for col in base_features for lag in feature_lags]

    scaler = joblib.load(REPO_ROOT / "models" / "expanded_scaler.joblib")
    X_scaled = scaler.transform(df_test[flat_cols].values)
    y_true = df_test["is_failure_within_next_k"].values.astype(np.float32)

    mlp = SmallMLP(15, 32, 16)
    mlp.load_state_dict(torch.load(REPO_ROOT / "models" / "expanded_mlp.pt", map_location="cpu"))
    mlp.eval()

    with torch.no_grad():
        probs = torch.sigmoid(mlp(torch.tensor(X_scaled, dtype=torch.float32))).numpy()

    # Exact metrics
    auroc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)
    prevalence = float(y_true.mean())

    fpr, tpr, _ = roc_curve(y_true, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)

    # Operating point at locked threshold 0.50
    y_pred_50 = (probs >= 0.50).astype(int)
    cm = confusion_matrix(y_true, y_pred_50)
    tn, fp, fn, tp = cm.ravel()
    opr_fpr = fp / (fp + tn)
    opr_tpr = tp / (tp + fn)
    opr_prec = tp / (tp + fp)
    opr_rec = opr_tpr
    opr_f1 = f1_score(y_true, y_pred_50)

    # Create 3-panel figure with clean horizontal breathing room
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300)
    fig.subplots_adjust(wspace=0.36, top=0.88, bottom=0.18, left=0.08, right=0.98)

    # Panel (a): ROC Curve
    ax_roc = axes[0]
    ax_roc.plot(fpr, tpr, color=NAVY, lw=1.8, label=f"SmallMLP (AUROC = {auroc:.4f})")
    ax_roc.fill_between(fpr, tpr, alpha=0.08, color=NAVY)
    ax_roc.plot([0, 1], [0, 1], color="#7f8c8d", linestyle=":", lw=1.1, label="Chance (0.5000)")
    ax_roc.scatter([opr_fpr], [opr_tpr], color=RED, s=32, zorder=6,
                   label=f"Locked θ=0.50 (TPR={opr_tpr:.3f})")

    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_xlabel("False Positive Rate (1 - Spec.)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("(a) ROC Curve", fontweight="bold", pad=6)
    ax_roc.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor=BORDER_GRAY, fontsize=6.8)

    # Panel (b): Precision-Recall Curve
    ax_pr = axes[1]
    ax_pr.plot(recall_curve, precision_curve, color=PURPLE, lw=1.8, label=f"SmallMLP (AUPRC = {auprc:.4f})")
    ax_pr.fill_between(recall_curve, precision_curve, alpha=0.08, color=PURPLE)
    ax_pr.axhline(prevalence, color="#7f8c8d", linestyle=":", lw=1.1, label=f"Prevalence ({prevalence:.3f})")
    ax_pr.scatter([opr_rec], [opr_prec], color=RED, s=32, zorder=6,
                  label=f"Locked θ=0.50 (F1={opr_f1:.3f})")

    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.set_xlabel("Recall (Sensitivity)")
    ax_pr.set_ylabel("Precision (PPV)")
    ax_pr.set_title("(b) Precision-Recall Curve", fontweight="bold", pad=6)
    ax_pr.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor=BORDER_GRAY, fontsize=6.8)

    # Panel (c): Decision Threshold Sensitivity
    ax_th = axes[2]
    thresholds = np.linspace(0.10, 0.90, 81)
    f1_vals, prec_vals, rec_vals = [], [], []
    for th in thresholds:
        yp = (probs >= th).astype(int)
        f1_vals.append(f1_score(y_true, yp, zero_division=0))
        prec_vals.append(precision_score(y_true, yp, zero_division=0))
        rec_vals.append(recall_score(y_true, yp, zero_division=0))

    ax_th.plot(thresholds, f1_vals, color=GREEN, lw=1.6, label="F1 Score")
    ax_th.plot(thresholds, prec_vals, color=BLUE, lw=1.4, linestyle="-.", label="Precision")
    ax_th.plot(thresholds, rec_vals, color=AMBER, lw=1.4, linestyle="--", label="Recall")
    ax_th.axvline(0.50, color=RED, linestyle=":", lw=1.2, label="Locked θ = 0.50")
    ax_th.scatter([0.50], [opr_f1], color=RED, s=30, zorder=6)

    ax_th.set_xlim(0.08, 0.92)
    ax_th.set_ylim(-0.02, 1.02)
    ax_th.set_xlabel("Decision Threshold (θ)")
    ax_th.set_ylabel("Metric Value")
    ax_th.set_title("(c) Operating Threshold Trade-Off", fontweight="bold", pad=6)
    ax_th.legend(loc="lower center", frameon=True, framealpha=0.9, edgecolor=BORDER_GRAY, fontsize=6.8)

    fig_name = "fig2_held_out_performance"
    plt.savefig(OUTPUT_DIR / f"{fig_name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.svg", format="svg", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {fig_name} (.pdf, .svg, .png)")


# ---------------------------------------------------------------------------
# FIGURE 3: TEMPORAL EARLY-WARNING BEHAVIOR
# ---------------------------------------------------------------------------
def generate_figure_3():
    """Figure 3: Temporal early-warning behavior for representative episode (sweep_A_C_R1)."""
    print("\n[Fig 3] Generating temporal early-warning behavior figure...")

    # Load flight data
    data_dir = REPO_ROOT / "data" / "raw" / "sweep_A_C_R1"
    gt = pd.read_csv(data_dir / "dataset_gt.csv")
    raw = pd.read_csv(data_dir / "raw_vo.csv")

    active_idx = gt["pos_z"].astype(float) >= 2.0
    t0 = gt.loc[active_idx, "timestamp_total_sec"].astype(float).min()
    t1 = gt.loc[active_idx, "timestamp_total_sec"].astype(float).max()
    in_window = (raw["timestamp_total_sec"] >= t0) & (raw["timestamp_total_sec"] <= t1)
    df = raw.loc[in_window].copy().reset_index(drop=True)

    # Causal replay inference
    from scripts.streaming_failure_predictor import StreamingFailurePredictor
    predictor = StreamingFailurePredictor(
        REPO_ROOT / "models" / "expanded_scaler.joblib",
        REPO_ROOT / "models" / "expanded_mlp.pt",
        threshold=0.50
    )
    predictor.reset()

    rows = []
    for i, r in df.iterrows():
        res = predictor.update({
            "timestamp": float(r["timestamp_total_sec"]),
            "eis_yaw_rate_deg": float(r["eis_yaw_rate_deg"]),
            "feature_vel_mean": float(r["feature_vel_mean"]),
            "is_r_frame": int(r["is_r_frame"]),
        })
        inliers = int(r["num_inliers_pose"])
        rows.append({
            "idx": i,
            "t": float(r["timestamp_total_sec"]),
            "yaw": float(r["eis_yaw_rate_deg"]),
            "flow": float(r["feature_vel_mean"]),
            "r_frame": int(r["is_r_frame"]),
            "prob": res["prob"],
            "pred": res["pred_label"],
            "inliers": inliers,
            "fail": int(inliers < 8)
        })
    df_res = pd.DataFrame(rows)

    # Representative Episode: Episode 40 (t = 28.45 s to 29.05 s)
    t_start_plot = 28.45
    t_end_plot = 29.05
    sub = df_res[(df_res["t"] >= t_start_plot) & (df_res["t"] <= t_end_plot)].copy()

    # Exact event timestamps:
    # Warning onset: t = 28.744 s (crosses threshold 0.50)
    # Failure onset: t = 28.909 s (inliers drop to 0 < 8)
    # Lead time = 0.165 s (5 frames @ 30 Hz)
    t_warn_onset = 28.744
    t_fail_onset = 28.909
    t_fail_end = 28.942

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.4), sharex=True, dpi=300,
                             gridspec_kw={"height_ratios": [1.0, 1.1, 1.0], "hspace": 0.28})

    # Shading regions common across all subplots
    for ax in axes:
        ax.axvspan(t_warn_onset, t_fail_onset, color="#f9e79f", alpha=0.45, zorder=1)
        ax.axvspan(t_fail_onset, t_fail_end, color="#f5b7b1", alpha=0.55, zorder=1)
        ax.axvline(t_warn_onset, color=AMBER, linestyle="--", lw=1.0, zorder=2)
        ax.axvline(t_fail_onset, color=RED, linestyle="--", lw=1.0, zorder=2)

    # Subplot (a): Telemetry Streams
    ax_tel = axes[0]
    ax_flow = ax_tel.twinx()

    l1, = ax_tel.plot(sub["t"], sub["yaw"], color=NAVY, lw=1.5, marker="o", markersize=3.2, label="Yaw Rate, ω_z (°/s)")
    l2, = ax_flow.plot(sub["t"], sub["flow"], color=TEAL, lw=1.3, linestyle="-.", marker="^", markersize=3.2, label="Optical Flow, v̄_flow (px)")

    ax_tel.set_ylabel("Yaw Rate (°/s)", color=NAVY)
    ax_flow.set_ylabel("Flow (px)", color=TEAL)
    ax_tel.tick_params(axis="y", labelcolor=NAVY)
    ax_flow.tick_params(axis="y", labelcolor=TEAL)
    ax_tel.set_ylim(25, 60)
    ax_flow.set_ylim(-2, 28)
    ax_tel.set_title("(a) Onboard Flight Telemetry Streams (30 Hz)", fontweight="bold", pad=5)

    ax_tel.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left",
                  frameon=True, framealpha=0.9, edgecolor=BORDER_GRAY, fontsize=6.8)

    # Subplot (b): Model Predicted Failure Probability
    ax_prob = axes[1]
    ax_prob.plot(sub["t"], sub["prob"], color=PURPLE, lw=1.7, marker="s", markersize=3.5, label="P(fail_{t+5} | x_t)")
    ax_prob.axhline(0.50, color="#7f8c8d", linestyle=":", lw=1.1, label="Decision Threshold (θ = 0.50)")

    # Lead time annotation arrow
    ax_prob.annotate(
        "", xy=(t_fail_onset, 0.78), xytext=(t_warn_onset, 0.78),
        arrowprops=dict(arrowstyle="<->", color=DARK_SLATE, lw=1.2)
    )
    ax_prob.text((t_warn_onset + t_fail_onset) / 2, 0.82, "Lead Time\nΔt = 165 ms",
                 ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=DARK_SLATE)

    ax_prob.text(t_warn_onset - 0.005, 0.54, "Warning Onset\n(θ crossed)",
                 ha="right", va="bottom", fontsize=6.8, color=AMBER, fontweight="semibold")

    ax_prob.set_ylabel("Failure Prob, P")
    ax_prob.set_ylim(0.20, 1.05)
    ax_prob.set_title("(b) Imminent Failure Anticipation & Warning State", fontweight="bold", pad=5)
    ax_prob.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor=BORDER_GRAY, fontsize=6.8)

    # Subplot (c): Ground Truth Visual Odometry State
    ax_vo = axes[2]
    ax_vo.plot(sub["t"], sub["inliers"], color=DARK_SLATE, lw=1.5, marker="D", markersize=3.2, label="VO Inlier Count (num_inliers_pose)")
    ax_vo.axhline(8, color=RED, linestyle="--", lw=1.1, label="VO Health Threshold (N = 8)")

    ax_vo.text(t_fail_onset + 0.015, 20, "Collapse Onset\n(Inliers = 0 < 8)",
               ha="left", va="bottom", fontsize=6.8, color=RED, fontweight="semibold")

    ax_vo.text(t_warn_onset + 0.01, 230, "Healthy Tracking\nDuring Warning Window\n(Inliers = 84–144 > 8)",
               ha="left", va="center", fontsize=6.8, color="#2c3e50", fontstyle="italic")

    ax_vo.set_xlabel("Flight Timestamp, t (seconds)")
    ax_vo.set_ylabel("Inlier Count")
    ax_vo.set_ylim(-30, 480)
    ax_vo.set_title("(c) Visual Odometry Solver Tracking State", fontweight="bold", pad=5)
    ax_vo.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor=BORDER_GRAY, fontsize=6.8)

    ax_vo.set_xlim(t_start_plot, t_end_plot)

    fig_name = "fig3_temporal_early_warning"
    plt.savefig(OUTPUT_DIR / f"{fig_name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.svg", format="svg", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {fig_name} (.pdf, .svg, .png)")


# ---------------------------------------------------------------------------
# FIGURE 4: CROSS-FLIGHT GENERALIZATION ACROSS 14 HELD-OUT TEST FLIGHTS
# ---------------------------------------------------------------------------
def generate_figure_4():
    """Figure 4: Generalization across 14 held-out test flights."""
    print("\n[Fig 4] Generating cross-flight generalization figure...")

    with open(REPO_ROOT / "models" / "expanded_evaluation_metrics.json") as f:
        data = json.load(f)

    per_flight = data["per_flight_test_k5"]
    pooled_auc = data["k5_primary"]["auroc"]
    pooled_f1 = data["k5_primary"]["f1"]
    pooled_prec = data["k5_primary"]["precision"]
    pooled_rec = data["k5_primary"]["recall"]

    flights = [item["flight"].replace("sweep_", "") for item in per_flight]
    aurocs = [item["auroc"] for item in per_flight]
    f1s = [item["f1"] for item in per_flight]
    precs = [item["precision"] for item in per_flight]
    recs = [item["recall"] for item in per_flight]

    # Sort flights by AUROC
    sort_idx = np.argsort(aurocs)
    flights_sorted = [flights[i] for i in sort_idx]
    aurocs_sorted = [aurocs[i] for i in sort_idx]
    f1s_sorted = [f1s[i] for i in sort_idx]
    precs_sorted = [precs[i] for i in sort_idx]
    recs_sorted = [recs[i] for i in sort_idx]

    mean_auc = float(np.mean(aurocs))

    # Color code by commanded yaw dynamics
    def get_color(fl):
        if fl.startswith("G"):
            return GREEN
        elif fl.startswith("M"):
            return BLUE
        elif fl.startswith("A"):
            return AMBER
        elif fl.startswith("E"):
            return RED
        return DARK_SLATE

    colors_sorted = [get_color(fl) for fl in flights_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.0), dpi=300,
                                   gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.22})
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.12, right=0.98)

    y = np.arange(len(flights_sorted))

    # Panel (a): Ranked AUROC point / stem plot
    ax1.hlines(y, 0.45, aurocs_sorted, color="#bdc3c7", lw=1.1, zorder=2)
    for i, (yy, score, col) in enumerate(zip(y, aurocs_sorted, colors_sorted)):
        ax1.scatter([score], [yy], color=col, s=36, zorder=4, edgecolor="#2c3e50", lw=0.6)
        ax1.text(score + 0.008, yy, f"{score:.3f}", va="center", fontsize=7.2, color="#222222")

    # Benchmark lines
    ax1.axvline(pooled_auc, color=RED, linestyle="--", lw=1.3, label=f"Pooled Test AUROC ({pooled_auc:.4f})")
    ax1.axvline(mean_auc, color=ORANGE, linestyle=":", lw=1.3, label=f"Mean Flight AUROC ({mean_auc:.4f})")
    ax1.axvline(0.50, color="#7f8c8d", linestyle="-", lw=1.0, label="Chance Level (0.5000)")

    ax1.set_yticks(y)
    ax1.set_yticklabels(flights_sorted, fontsize=7.6)
    ax1.set_xlim(0.44, 0.92)
    ax1.set_xlabel("Held-Out Test AUROC")
    ax1.set_title("(a) AUROC Across 14 Disjoint Test Flights", fontweight="bold", pad=6)

    # Single consolidated legend in upper left (unobstructed by data points)
    combined_handles = [
        plt.Line2D([0], [0], color=RED, linestyle="--", lw=1.3, label=f"Pooled Test ({pooled_auc:.4f})"),
        plt.Line2D([0], [0], color=ORANGE, linestyle=":", lw=1.3, label=f"Mean Flight ({mean_auc:.4f})"),
        plt.Line2D([0], [0], color="#7f8c8d", linestyle="-", lw=1.0, label="Chance (0.5000)"),
        patches.Patch(facecolor=GREEN, edgecolor="#2c3e50", label="Gentle (G, 7.5°/s)"),
        patches.Patch(facecolor=BLUE, edgecolor="#2c3e50", label="Moderate (M, 22.5°/s)"),
        patches.Patch(facecolor=AMBER, edgecolor="#2c3e50", label="Aggressive (A, 45°/s)"),
        patches.Patch(facecolor=RED, edgecolor="#2c3e50", label="Extreme (E, 90°/s)"),
    ]
    # Consolidated legend in empty upper-left region of panel (a) (x <= 0.62, unobstructed from markers at x >= 0.80)
    ax1.legend(handles=combined_handles, loc="upper left", fontsize=5.8,
               handlelength=1.1, handletextpad=0.35, borderpad=0.25, labelspacing=0.25,
               frameon=True, framealpha=0.92, edgecolor=BORDER_GRAY)

    # Panel (b): Per-Flight Precision, Recall, and F1 Profiles
    bar_width = 0.26
    ax2.barh(y - bar_width, precs_sorted, height=bar_width, color=BLUE, alpha=0.85, label="Precision")
    ax2.barh(y, f1s_sorted, height=bar_width, color=GREEN, alpha=0.85, label="F1 Score")
    ax2.barh(y + bar_width, recs_sorted, height=bar_width, color=AMBER, alpha=0.85, label="Recall")

    # Pooled aggregate line references
    ax2.axvline(pooled_f1, color=GREEN, linestyle="--", lw=1.1, alpha=0.7, label=f"Pooled F1 ({pooled_f1:.3f})")
    ax2.axvline(pooled_prec, color=BLUE, linestyle=":", lw=1.1, alpha=0.7, label=f"Pooled Prec ({pooled_prec:.3f})")
    ax2.axvline(pooled_rec, color=AMBER, linestyle="-.", lw=1.1, alpha=0.7, label=f"Pooled Rec ({pooled_rec:.3f})")

    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0.0, 1.05)
    ax2.set_xlabel("Metric Value (Locked θ = 0.50)")
    ax2.set_title("(b) Precision, Recall & F1 Trade-Off", fontweight="bold", pad=6)
    ax2.legend(loc="lower right", fontsize=6.8, frameon=True, framealpha=0.9, edgecolor=BORDER_GRAY)

    fig_name = "fig4_cross_flight_generalization"
    plt.savefig(OUTPUT_DIR / f"{fig_name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.svg", format="svg", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {fig_name} (.pdf, .svg, .png)")


# ---------------------------------------------------------------------------
# FIGURE 5: REAL-TIME FEASIBILITY & ROS 2 MIDDLEWARE VALIDATION
# ---------------------------------------------------------------------------
def generate_figure_5():
    """Figure 5: Real-time computational feasibility and live ROS 2 middleware latency."""
    print("\n[Fig 5] Generating real-time feasibility figure...")

    # Load causal replay metrics and live ROS 2 predictions
    with open(REPO_ROOT / "data" / "processed" / "causal_replay_evaluation_metrics.json") as f:
        rep_data = json.load(f)

    with open(REPO_ROOT / "data" / "processed" / "live_ros2_validation_summary.json") as f:
        live_sum = json.load(f)

    df_live = pd.read_csv(REPO_ROOT / "data" / "processed" / "live_ros2_predictions.csv")
    valid_live = df_live[df_live["has_prediction"] == 1].copy()
    live_lat_ms = valid_live["e2e_lat_us"].values / 1000.0  # us to ms

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=300,
                             gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    fig.subplots_adjust(wspace=0.38, top=0.88, bottom=0.18, left=0.14, right=0.98)

    # Panel (a): Algorithmic Stage Latency Breakdown (Microseconds)
    ax_stage = axes[0]
    stages = ["Feature FIFO", "StandardScaler", "SmallMLP Forward", "Total Algorithmic"]
    lat_means_us = [
        rep_data["latency_report"]["feature_construction"]["mean_us"],
        rep_data["latency_report"]["scaler_transform"]["mean_us"],
        rep_data["latency_report"]["mlp_inference"]["mean_us"],
        rep_data["latency_report"]["end_to_end"]["mean_us"],
    ]
    lat_p99_us = [
        rep_data["latency_report"]["feature_construction"]["p99_us"],
        rep_data["latency_report"]["scaler_transform"]["p99_us"],
        rep_data["latency_report"]["mlp_inference"]["p99_us"],
        rep_data["latency_report"]["end_to_end"]["p99_us"],
    ]

    y_s = np.arange(len(stages))
    bar_h = 0.30
    ax_stage.barh(y_s - 0.16, lat_means_us, height=bar_h, color=BLUE, alpha=0.85, label="Mean Latency")
    ax_stage.barh(y_s + 0.16, lat_p99_us, height=bar_h, color=NAVY, alpha=0.85, label="P99 Latency")

    # Unambiguously annotate both Mean and P99 bars
    for i, (m, p) in enumerate(zip(lat_means_us, lat_p99_us)):
        if i == 3:
            # Total Algorithmic: highlight headline mean (0.128 ms) and P99 (0.214 ms)
            ax_stage.text(m + 5, i - 0.16, f"Mean: {m:.1f} μs (0.128 ms)", va="center", fontsize=5.9, color=BLUE, fontweight="bold")
            ax_stage.text(p + 5, i + 0.16, f"P99: {p:.1f} μs (0.214 ms)", va="center", fontsize=5.9, color=NAVY, fontweight="bold")
        else:
            ax_stage.text(m + 5, i - 0.16, f"{m:.1f} μs", va="center", fontsize=6.2, color=BLUE)
            ax_stage.text(p + 5, i + 0.16, f"{p:.1f} μs", va="center", fontsize=6.2, color=NAVY)

    ax_stage.set_yticks(y_s)
    ax_stage.set_yticklabels(stages, fontsize=7.2)
    ax_stage.set_xlabel("Latency (μs)")
    ax_stage.set_xlim(0, 360)
    ax_stage.set_title("(a) Algorithmic Stage Latency", fontweight="bold", pad=6)
    ax_stage.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.92, edgecolor=BORDER_GRAY)

    # Panel (b): Live ROS 2 Middleware Latency Distribution
    ax_dist = axes[1]
    clipped_lats = np.clip(live_lat_ms, 0, 5.0)

    n_bins = 40
    counts, bins, patches_list = ax_dist.hist(clipped_lats, bins=n_bins, color=PURPLE,
                                              alpha=0.75, edgecolor="#ffffff", lw=0.5, density=True)

    mean_live = live_sum["latency_live"]["mean_ms"]
    med_live = live_sum["latency_live"]["median_ms"]
    p95_live = live_sum["latency_live"]["p95_ms"]
    p99_live = live_sum["latency_live"]["p99_ms"]

    ax_dist.axvline(med_live, color=GREEN, lw=1.3, linestyle="-", label=f"Median: {med_live:.2f} ms")
    ax_dist.axvline(mean_live, color=BLUE, lw=1.3, linestyle="--", label=f"Mean: {mean_live:.2f} ms")
    ax_dist.axvline(p99_live, color=RED, lw=1.3, linestyle=":", label=f"P99: {p99_live:.2f} ms")

    ax_dist.text(0.96, 0.92, f"30 Hz Budget: 33.3 ms\nP99 Uses: 11.1%\nHeadroom: >88.8%",
                 transform=ax_dist.transAxes, ha="right", va="top",
                 fontsize=6.8, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7", edgecolor=BORDER_GRAY, alpha=0.9))

    ax_dist.set_xlim(0, 4.5)
    ax_dist.set_xlabel("Execution Latency (ms)")
    ax_dist.set_ylabel("Probability Density")
    ax_dist.set_title("(b) ROS 2 Execution Latency", fontweight="bold", pad=6)
    ax_dist.legend(loc="center right", fontsize=6.8, frameon=True, framealpha=0.9, edgecolor=BORDER_GRAY)

    # Panel (c): Operational Timing Budget & Headroom
    ax_budget = axes[2]

    rates = [30.0, 192.0]
    x_r = np.array([0.3, 0.8])
    ax_budget.bar(x_r, rates, width=0.32, color=[NAVY, TEAL], alpha=0.85, edgecolor="#2c3e50", lw=0.8)

    ax_budget.text(0.3, 30 + 7, "30 Hz", ha="center", fontsize=7.2, fontweight="bold", color=NAVY)
    ax_budget.text(0.8, 192 + 7, "192 Hz\n(6.4× Margin)", ha="center", fontsize=7.2, fontweight="bold", color=TEAL)

    ax_budget.set_xticks(x_r)
    ax_budget.set_xticklabels(["Required\nStream", "Sustained\nCapacity"], fontsize=7.4)
    ax_budget.set_ylabel("Throughput Rate (Hz)")
    ax_budget.set_ylim(0, 255)
    ax_budget.set_title("(c) Throughput & Margin", fontweight="bold", pad=6)

    fig_name = "fig5_realtime_feasibility"
    plt.savefig(OUTPUT_DIR / f"{fig_name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.svg", format="svg", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"{fig_name}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved {fig_name} (.pdf, .svg, .png)")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Research 2 — Publication Figure Generation Suite")
    print(f"Destination: {OUTPUT_DIR}")
    print("=" * 70)

    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()

    print("\n" + "=" * 70)
    print("SUCCESS: All 5 publication figures successfully generated.")
    print("Formats produced: PDF (vector), SVG (vector), PNG (300 DPI)")
    print("=" * 70)


if __name__ == "__main__":
    main()
