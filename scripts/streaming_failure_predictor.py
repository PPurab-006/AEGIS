#!/usr/bin/env python3
"""
Research 2 — Causal Streaming Inference Engine for Imminent VO Failure Prediction.

Maintains a rolling telemetry history and executes strictly causal, sample-by-sample
failure prediction using the frozen expanded Small MLP model and fitted scaler.

CRITICAL CONSTRAINTS:
  - Only uses information available at or before the current timestamp.
  - Zero future data access (no future rows, no negative offsets).
  - Scaler is loaded pre-fitted (models/expanded_scaler.joblib); never refit.
  - Model weights loaded pre-trained (models/expanded_mlp.pt); never modified.
  - Locked decision threshold: 0.5.
  - Measures individual stage execution latencies (Feature construction, Scaler, MLP, End-to-end).
"""

import collections
import time
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """Locked 2-hidden-layer MLP architecture (15 -> 32 -> 16 -> 1)."""

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


BASE_FEATURES = ["eis_yaw_rate_deg", "feature_vel_mean", "is_r_frame"]
FEATURE_LAGS = [0, 1, 2, 3, 5]
FEATURE_COLS = [f"{col}_lag{lag}" for col in BASE_FEATURES for lag in FEATURE_LAGS]


class StreamingFailurePredictor:
    """Causal, rolling-buffer streaming inference engine for live failure prediction."""

    BUFFER_SIZE = 6  # 6 samples required to provide lag 0 (index -1) through lag 5 (index -6)

    def __init__(
        self,
        scaler_path: Path,
        model_path: Path,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.threshold = threshold
        self.device = torch.device(device)

        # 1. Load frozen fitted scaler
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # 2. Load frozen model weights
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        self.model = SmallMLP(input_dim=15, hidden1=32, hidden2=16)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 3. Rolling telemetry history buffer (maxlen=6)
        self.buffer = collections.deque(maxlen=self.BUFFER_SIZE)
        self.total_samples_received = 0

    def reset(self):
        """Reset rolling buffer (e.g., at the start of a new flight or active window)."""
        self.buffer.clear()
        self.total_samples_received = 0

    def update(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a single telemetry sample at the current instant and produce a prediction.

        Parameters:
            telemetry: Dict with keys:
                - 'timestamp': current timestamp (seconds)
                - 'eis_yaw_rate_deg': absolute yaw rate in deg/s
                - 'feature_vel_mean': optical flow magnitude in px/frame
                - 'is_r_frame': binary flag for yaw rate > 15 deg/s

        Returns:
            Dict containing:
                - 'timestamp': input timestamp
                - 'has_prediction': bool (True once 6 historical samples exist)
                - 'prob': float prediction probability
                - 'pred_label': int binary prediction (0 or 1)
                - 'latencies_us': dict with individual stage latencies in microseconds
                - 'features_unscaled': dict of the 15 unscaled feature values
        """
        t_start = time.perf_counter()

        ts = float(telemetry.get("timestamp", 0.0))
        sample = {
            "timestamp": ts,
            "eis_yaw_rate_deg": float(telemetry.get("eis_yaw_rate_deg", 0.0)),
            "feature_vel_mean": float(telemetry.get("feature_vel_mean", 0.0)),
            "is_r_frame": int(telemetry.get("is_r_frame", 0)),
        }

        # Causal rolling buffer append: past samples shift automatically
        self.buffer.append(sample)
        self.total_samples_received += 1

        # Check warmup boundary (first 5 frames lack a full 5-lag history)
        if len(self.buffer) < self.BUFFER_SIZE:
            t_end = time.perf_counter()
            return {
                "timestamp": ts,
                "has_prediction": False,
                "prob": float("nan"),
                "pred_label": -1,
                "latencies_us": {
                    "feature_construction_us": 0.0,
                    "scaler_transform_us": 0.0,
                    "mlp_inference_us": 0.0,
                    "end_to_end_us": (t_end - t_start) * 1e6,
                },
                "features_unscaled": None,
            }

        # --- Stage A: Feature Construction ---
        t_feat_start = time.perf_counter()
        # buffer[-1] is lag0 (current)
        # buffer[-2] is lag1 (1 frame prior)
        # buffer[-3] is lag2 (2 frames prior)
        # buffer[-4] is lag3 (3 frames prior)
        # buffer[-6] is lag5 (5 frames prior)
        features_unscaled = {
            "eis_yaw_rate_deg_lag0": self.buffer[-1]["eis_yaw_rate_deg"],
            "eis_yaw_rate_deg_lag1": self.buffer[-2]["eis_yaw_rate_deg"],
            "eis_yaw_rate_deg_lag2": self.buffer[-3]["eis_yaw_rate_deg"],
            "eis_yaw_rate_deg_lag3": self.buffer[-4]["eis_yaw_rate_deg"],
            "eis_yaw_rate_deg_lag5": self.buffer[-6]["eis_yaw_rate_deg"],
            "feature_vel_mean_lag0": self.buffer[-1]["feature_vel_mean"],
            "feature_vel_mean_lag1": self.buffer[-2]["feature_vel_mean"],
            "feature_vel_mean_lag2": self.buffer[-3]["feature_vel_mean"],
            "feature_vel_mean_lag3": self.buffer[-4]["feature_vel_mean"],
            "feature_vel_mean_lag5": self.buffer[-6]["feature_vel_mean"],
            "is_r_frame_lag0": self.buffer[-1]["is_r_frame"],
            "is_r_frame_lag1": self.buffer[-2]["is_r_frame"],
            "is_r_frame_lag2": self.buffer[-3]["is_r_frame"],
            "is_r_frame_lag3": self.buffer[-4]["is_r_frame"],
            "is_r_frame_lag5": self.buffer[-6]["is_r_frame"],
        }
        raw_vec = np.array([features_unscaled[c] for c in FEATURE_COLS], dtype=np.float64).reshape(1, -1)
        t_feat_end = time.perf_counter()

        # --- Stage B: Scaler Transformation ---
        t_scale_start = time.perf_counter()
        scaled_vec = self.scaler.transform(raw_vec)
        t_scale_end = time.perf_counter()

        # --- Stage C: MLP Inference ---
        t_mlp_start = time.perf_counter()
        with torch.no_grad():
            x_tensor = torch.tensor(scaled_vec, dtype=torch.float32, device=self.device)
            logit = self.model(x_tensor)
            prob = float(torch.sigmoid(logit).item())
            pred_label = 1 if prob >= self.threshold else 0
        t_mlp_end = time.perf_counter()

        t_end = time.perf_counter()

        return {
            "timestamp": ts,
            "has_prediction": True,
            "prob": prob,
            "pred_label": pred_label,
            "latencies_us": {
                "feature_construction_us": (t_feat_end - t_feat_start) * 1e6,
                "scaler_transform_us": (t_scale_end - t_scale_start) * 1e6,
                "mlp_inference_us": (t_mlp_end - t_mlp_start) * 1e6,
                "end_to_end_us": (t_end - t_start) * 1e6,
            },
            "features_unscaled": features_unscaled,
        }
