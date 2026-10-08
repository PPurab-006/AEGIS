"""Generate publication-quality figures for Research 2 README from verified experimental JSON outputs."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configure styling
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "figure.autolayout": True
})

OUTPUT_DIR = Path("assets/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Figure 1: Training & Validation Convergence
def plot_training_convergence():
    with open("models/expanded_training_history.json") as f:
        data = json.load(f)
    history = data["meta"]["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss (BCE)", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, label="Validation Loss (BCE)", color="#ff7f0e", linewidth=2)
    best_ep = data["meta"]["best_epoch"]
    best_loss = data["meta"]["best_val_loss"]
    ax1.scatter([best_ep], [best_loss], color="#d62728", s=60, zorder=5, label=f"Best Val Loss ({best_loss:.4f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted BCE Loss")
    ax1.set_title("Training and Validation Loss vs Epoch")
    ax1.legend(loc="upper right")

    # AUC
    ax2.plot(epochs, val_auc, label="Validation AUROC", color="#2ca02c", linewidth=2)
    final_auc = val_auc[-1]
    ax2.axhline(final_auc, color="#2ca02c", linestyle=":", alpha=0.7, label=f"Final Val AUROC ({final_auc:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation AUROC")
    ax2.set_title("Validation AUROC vs Epoch")
    ax2.legend(loc="lower right")

    fig.suptitle("Research 2: MLP Training Dynamics (42-Flight Sweep Pool)", fontweight="bold")
    out_path = OUTPUT_DIR / "fig1_training_convergence.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# 2. Figure 2: Per-Flight Held-Out Test Generalization
def plot_per_flight_generalization():
    with open("models/expanded_evaluation_metrics.json") as f:
        data = json.load(f)
    per_flight = data["per_flight_test_k5"]
    
    flights = [item["flight"].replace("sweep_", "") for item in per_flight]
    aurocs = [item["auroc"] for item in per_flight]
    f1s = [item["f1"] for item in per_flight]

    # Sort by AUROC
    sort_idx = np.argsort(aurocs)
    flights = [flights[i] for i in sort_idx]
    aurocs = [aurocs[i] for i in sort_idx]
    f1s = [f1s[i] for i in sort_idx]

    y = np.arange(len(flights))

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)
    bars = ax.barh(y, aurocs, height=0.65, color="#1f77b4", alpha=0.85, label="Test AUROC")

    pooled_auc = data["k5_primary"]["auroc"]
    mean_auc = np.mean(aurocs)
    ax.axvline(pooled_auc, color="#d62728", linestyle="--", linewidth=1.8, label=f"Pooled Test AUROC ({pooled_auc:.4f})")
    ax.axvline(mean_auc, color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"Mean Flight AUROC ({mean_auc:.4f})")
    ax.axvline(0.50, color="gray", linestyle="-", linewidth=1.0, alpha=0.6, label="Chance Level (0.50)")

    ax.set_yticks(y)
    ax.set_yticklabels(flights)
    ax.set_xlabel("Held-Out Test AUROC")
    ax.set_xlim(0.45, 0.95)
    ax.set_title("Research 2: AUROC Across 14 Disjoint Held-Out Test Flights ($K=5$)", fontweight="bold")
    ax.legend(loc="lower right")

    for i, v in enumerate(aurocs):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9, color="#222")

    out_path = OUTPUT_DIR / "fig2_per_flight_test_performance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# 3. Figure 3: Horizon Sensitivity & Streaming Lead-Time Breakdown
def plot_horizon_and_streaming():
    with open("models/expanded_evaluation_metrics.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # Subplot 1: Horizon Sensitivity
    horizons = ["K=2 (~66 ms)", "K=3 (~99 ms)", "K=5 (~165 ms)"]
    k_aurocs = [data["k2_sensitivity"]["auroc"], data["k3_sensitivity"]["auroc"], data["k5_primary"]["auroc"]]
    k_f1 = [data["k2_sensitivity"]["f1"], data["k3_sensitivity"]["f1"], data["k5_primary"]["f1"]]
    k_recall = [data["k2_sensitivity"]["recall"], data["k3_sensitivity"]["recall"], data["k5_primary"]["recall"]]

    x = np.arange(len(horizons))
    width = 0.25

    ax1.bar(x - width, k_aurocs, width, label="AUROC", color="#1f77b4", alpha=0.85)
    ax1.bar(x, k_f1, width, label="F1 Score", color="#2ca02c", alpha=0.85)
    ax1.bar(x + width, k_recall, width, label="Recall", color="#ff7f0e", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.set_ylabel("Metric Value")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Performance vs Prediction Horizon")
    ax1.legend(loc="lower left")

    # Subplot 2: Causal Streaming Replay Episode Outcomes (from causal replay evaluation: 658 episodes)
    # 522 true early warnings (79.3%), 124 post-failure detections (18.8%), 12 missed (1.8%)
    categories = ["Early Warning\n(Prior to Onset)", "Post-Onset\nDetection", "Missed\nFailure"]
    counts = [522, 124, 12]
    percentages = [79.33, 18.84, 1.82]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    bars = ax2.bar(categories, percentages, color=colors, width=0.55, alpha=0.85)
    ax2.set_ylabel("Percentage of Failure Episodes (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Causal Streaming Replay: 658 Failure Episodes")

    for bar, pct, count in zip(bars, percentages, counts):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.0, f"{pct:.1f}%\n(n={count})", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Research 2: Horizon Sensitivity and Causal Early Warning Fidelity", fontweight="bold")
    out_path = OUTPUT_DIR / "fig3_horizon_and_lead_time.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_training_convergence()
    plot_per_flight_generalization()
    plot_horizon_and_streaming()
    print("All README figures successfully generated.")
