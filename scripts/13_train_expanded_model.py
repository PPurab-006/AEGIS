#!/usr/bin/env python3
"""
Research 2 — Train Expanded Model on 42-Flight Sweep Pool (K=5).

Trains the locked Small MLP (15 -> 32 -> 16 -> 1) on the expanded dataset.

Locked Constraints:
  - Architecture: Small MLP (15 -> 32 -> 16 -> 1)
  - Criterion: BCEWithLogitsLoss(pos_weight=n_neg_train / n_pos_train)
  - Optimizer: Adam(lr=1e-3, weight_decay=1e-4)
  - Batch size: 64
  - Max epochs: 30
  - Early stopping: patience = 7, monitoring validation loss
  - Seed: 42
  - Scaler fit ONLY on training flights (models/expanded_scaler.joblib)
  - Model weights saved to models/expanded_mlp.pt
  - Test split is STRICTLY UNTOUCHED (quarantined until evaluation script)

Inputs:
  data/processed/expanded_frames_v2_k5.csv
  data/processed/expanded_flight_split.csv

Outputs:
  models/expanded_scaler.joblib
  models/expanded_mlp.pt
  models/expanded_training_history.json
"""

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": int(len(y_true)),
        "positives": int(y_true.sum()),
        "pos_rate": float(y_true.mean() * 100),
    }


def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    pos_weight: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
) -> tuple[nn.Module, dict]:
    """Train PyTorch model with early stopping on validation loss."""
    model.to(device)
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            v_logits = model(val_x)
            v_loss = criterion(v_logits, val_y).item()
            v_probs = torch.sigmoid(v_logits).cpu().numpy()
            v_auc = roc_auc_score(y_val, v_probs)

        mean_tr_loss = float(np.mean(train_losses))
        history.append({
            "epoch": epoch + 1,
            "train_loss": mean_tr_loss,
            "val_loss": float(v_loss),
            "val_auc": float(v_auc),
        })

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} (patience={patience}).")
                break

    model.load_state_dict(best_state)
    model.eval()
    meta = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "total_epochs": len(history),
        "history": history,
    }
    return model, meta


def main():
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "models"

    print("=================================================================")
    print("Research 2 — Training Expanded Small MLP Model (K=5 Horizon)")
    print("=================================================================")

    k5_path = processed_dir / "expanded_frames_v2_k5.csv"
    split_path = processed_dir / "expanded_flight_split.csv"

    if not k5_path.exists() or not split_path.exists():
        print("FATAL: Missing input datasets.", file=sys.stderr)
        sys.exit(1)

    df_k5 = pd.read_csv(k5_path)
    df_split = pd.read_csv(split_path)
    split_map = dict(zip(df_split["run_dir"], df_split["split"]))
    df_k5["split"] = df_k5["run_dir"].map(split_map)

    train_mask = df_k5["split"] == "train"
    val_mask = df_k5["split"] == "val"
    test_mask = df_k5["split"] == "test"

    print(f"Train set: {df_k5.loc[train_mask, 'run_dir'].nunique()} flights, {train_mask.sum():,} frames")
    print(f"Val set  : {df_k5.loc[val_mask, 'run_dir'].nunique()} flights, {val_mask.sum():,} frames")
    print(f"Test set : {df_k5.loc[test_mask, 'run_dir'].nunique()} flights, {test_mask.sum():,} frames (QUARANTINED)")

    # 1. Feature Scaling (Train-Only Fit)
    print("\nFitting StandardScaler on TRAIN split only...")
    scaler = StandardScaler()
    X_tr_flat = scaler.fit_transform(df_k5.loc[train_mask, FLAT_FEATURE_COLS].values)
    X_va_flat = scaler.transform(df_k5.loc[val_mask, FLAT_FEATURE_COLS].values)

    scaler_path = models_dir / "expanded_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Saved fitted scaler to: {scaler_path.name}")

    y_tr = df_k5.loc[train_mask, "is_failure_within_next_k"].values.astype(np.float32)
    y_va = df_k5.loc[val_mask, "is_failure_within_next_k"].values.astype(np.float32)

    n_pos_tr = y_tr.sum()
    n_neg_tr = len(y_tr) - n_pos_tr
    pos_weight_val = n_neg_tr / n_pos_tr
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"Class imbalance on TRAIN: pos={int(n_pos_tr):,}, neg={int(n_neg_tr):,} -> pos_weight={pos_weight_val:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 2. Train Small MLP
    print("\nTraining Small MLP (15 -> 32 -> 16 -> 1)...")
    torch.manual_seed(SEED)
    mlp_model = SmallMLP(input_dim=15, hidden1=32, hidden2=16)
    mlp_model, mlp_meta = train_pytorch_model(
        mlp_model,
        X_tr_flat,
        y_tr,
        X_va_flat,
        y_va,
        pos_weight=pos_weight,
        device=device,
        epochs=30,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        patience=7,
    )

    mlp_model.eval()
    with torch.no_grad():
        tr_logits = mlp_model(torch.tensor(X_tr_flat, dtype=torch.float32).to(device))
        tr_probs = torch.sigmoid(tr_logits).cpu().numpy()
        va_logits = mlp_model(torch.tensor(X_va_flat, dtype=torch.float32).to(device))
        va_probs = torch.sigmoid(va_logits).cpu().numpy()

    tr_metrics = compute_metrics(y_tr, tr_probs)
    va_metrics = compute_metrics(y_va, va_probs)

    print("\n--- Training Results ---")
    print(f"Best Epoch: {mlp_meta['best_epoch']} / {mlp_meta['total_epochs']}")
    print(f"Train: AUROC = {tr_metrics['auroc']:.4f} | F1 = {tr_metrics['f1']:.4f} | "
          f"Precision = {tr_metrics['precision']:.4f} | Recall = {tr_metrics['recall']:.4f}")
    print(f"Val  : AUROC = {va_metrics['auroc']:.4f} | F1 = {va_metrics['f1']:.4f} | "
          f"Precision = {va_metrics['precision']:.4f} | Recall = {va_metrics['recall']:.4f}")
    print(f"Val Confusion Matrix: TN={va_metrics['tn']}, FP={va_metrics['fp']}, FN={va_metrics['fn']}, TP={va_metrics['tp']}")

    # Save artifacts
    mlp_path = models_dir / "expanded_mlp.pt"
    torch.save(mlp_model.state_dict(), mlp_path)
    print(f"\nSaved model weights to: {mlp_path.name}")

    history_payload = {
        "meta": mlp_meta,
        "train_metrics": tr_metrics,
        "val_metrics": va_metrics,
        "pos_weight": float(pos_weight_val),
        "split_summary": {
            "train_flights": int(df_k5.loc[train_mask, "run_dir"].nunique()),
            "train_rows": int(train_mask.sum()),
            "train_pos": int(n_pos_tr),
            "val_flights": int(df_k5.loc[val_mask, "run_dir"].nunique()),
            "val_rows": int(val_mask.sum()),
            "val_pos": int(y_va.sum()),
            "test_flights": int(df_k5.loc[test_mask, "run_dir"].nunique()),
            "test_rows": int(test_mask.sum()),
        },
    }
    history_path = models_dir / "expanded_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"Saved training history to: {history_path.name}")


if __name__ == "__main__":
    main()
