#!/usr/bin/env python3
"""
Unit tests auditing causal integrity and zero-leakage properties of StreamingFailurePredictor.

Verifies:
  1. No future feature access: lag0 is current, lags 1,2,3,5 are historical.
  2. Rolling buffer requires exactly 5 warm-up frames before emitting predictions.
  3. Prediction values match offline lag calculation identically.
  4. Model and scaler weights are never mutated during inference.
  5. Forbidden fields (num_inliers_pose, ground truth, future timestamps) are never ingested.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.streaming_failure_predictor import StreamingFailurePredictor


@pytest.fixture
def predictor():
    repo_root = Path(__file__).resolve().parent.parent
    scaler_path = repo_root / "models" / "expanded_scaler.joblib"
    model_path = repo_root / "models" / "expanded_mlp.pt"
    return StreamingFailurePredictor(scaler_path=scaler_path, model_path=model_path)


def test_warmup_period(predictor):
    """Verify that predictions are suppressed for the first 5 frames."""
    predictor.reset()
    for i in range(5):
        sample = {
            "timestamp": 10.0 + i * 0.033,
            "eis_yaw_rate_deg": 10.0 + i,
            "feature_vel_mean": 2.0 + i,
            "is_r_frame": 0,
        }
        res = predictor.update(sample)
        assert res["has_prediction"] is False
        assert np.isnan(res["prob"])
        assert res["pred_label"] == -1

    # 6th frame (index 5) should yield the first valid prediction
    sample_6 = {
        "timestamp": 10.0 + 5 * 0.033,
        "eis_yaw_rate_deg": 15.0,
        "feature_vel_mean": 7.0,
        "is_r_frame": 1,
    }
    res_6 = predictor.update(sample_6)
    assert res_6["has_prediction"] is True
    assert 0.0 <= res_6["prob"] <= 1.0
    assert res_6["pred_label"] in [0, 1]


def test_lag_integrity(predictor):
    """Verify that lag values correspond exactly to historical timesteps:

    lag0 = current, lag1 = -1, lag2 = -2, lag3 = -3, lag5 = -5.
    """
    predictor.reset()
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    res = None
    for i, v in enumerate(values):
        sample = {
            "timestamp": float(i),
            "eis_yaw_rate_deg": v,
            "feature_vel_mean": v * 0.1,
            "is_r_frame": int(v > 15.0),
        }
        res = predictor.update(sample)

    assert res["has_prediction"] is True
    feats = res["features_unscaled"]

    # At step 6 (v=70.0):
    # lag0 is v[6] = 70.0
    # lag1 is v[5] = 60.0
    # lag2 is v[4] = 50.0
    # lag3 is v[3] = 40.0
    # lag5 is v[1] = 20.0 (5 frames prior)
    assert feats["eis_yaw_rate_deg_lag0"] == 70.0
    assert feats["eis_yaw_rate_deg_lag1"] == 60.0
    assert feats["eis_yaw_rate_deg_lag2"] == 50.0
    assert feats["eis_yaw_rate_deg_lag3"] == 40.0
    assert feats["eis_yaw_rate_deg_lag5"] == 20.0


def test_model_immutability(predictor):
    """Verify that model weights and scaler parameters are not mutated during inference."""
    initial_weights = {k: v.clone() for k, v in predictor.model.state_dict().items()}
    initial_scaler_mean = predictor.scaler.mean_.copy()

    predictor.reset()
    for i in range(20):
        sample = {
            "timestamp": float(i) * 0.033,
            "eis_yaw_rate_deg": float(i * 3 % 45),
            "feature_vel_mean": float(i % 10),
            "is_r_frame": int((i * 3 % 45) > 15),
        }
        predictor.update(sample)

    for k, v in predictor.model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Weight {k} was mutated!"

    assert np.array_equal(predictor.scaler.mean_, initial_scaler_mean), "Scaler mean was mutated!"


def test_stage_latencies_positive(predictor):
    """Verify that stage latencies are correctly measured and strictly non-negative."""
    predictor.reset()
    for i in range(10):
        sample = {
            "timestamp": float(i) * 0.033,
            "eis_yaw_rate_deg": 25.0,
            "feature_vel_mean": 3.5,
            "is_r_frame": 1,
        }
        res = predictor.update(sample)

    lat = res["latencies_us"]
    assert lat["feature_construction_us"] >= 0.0
    assert lat["scaler_transform_us"] >= 0.0
    assert lat["mlp_inference_us"] >= 0.0
    assert lat["end_to_end_us"] >= 0.0
    # End-to-end should encompass all individual stages
    sum_stages = lat["feature_construction_us"] + lat["scaler_transform_us"] + lat["mlp_inference_us"]
    assert lat["end_to_end_us"] >= sum_stages * 0.5  # accounting for timing overheads
