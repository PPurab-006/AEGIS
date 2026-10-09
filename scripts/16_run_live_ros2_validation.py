#!/usr/bin/env python3
"""
Research 2 — Part 9 & 10: Live ROS 2 Deployment Validation Runner.

Executes live validation using ROS 2 pub/sub communications:
  1. Launches LiveFailurePredictorNode (subscribing to /telemetry/motion).
  2. Publishes real-time motion telemetry from a validated flight condition
     (A_C: aggressive circle at 45 deg/s yaw rate, safe and produces measurable VO failures).
  3. Receives live predictions on /vo/failure_prediction.
  4. Records arrival times, stage latencies, online inference rate, and jitter.
  5. Synchronizes post-flight against VO failure ground truth to verify whether
     warnings occurred prior to failure onset under live ROS 2 middleware execution.

Inputs:
  models/expanded_scaler.joblib
  models/expanded_mlp.pt
  <AEGIS_DATA_DIR>/sweep_A_C_R1/

Outputs:
  data/processed/live_ros2_predictions.csv
  data/processed/live_ros2_validation_summary.json
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Vector3Stamped, PointStamped

from scripts.live_ros2_failure_predictor import LiveFailurePredictorNode


class LiveTelemetryStreamerNode(Node):
    """Publishes telemetry messages in real-time to simulate live vehicle sensor stream."""

    def __init__(self, telemetry_topic: str = "/telemetry/motion"):
        super().__init__("live_telemetry_streamer_node")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(Vector3Stamped, telemetry_topic, qos)

    def publish_sample(self, ts: float, yaw_rate: float, vel_mean: float, is_r: int):
        msg = Vector3Stamped()
        msg.header.stamp.sec = int(ts)
        msg.header.stamp.nanosec = int((ts - int(ts)) * 1e9)
        msg.vector.x = float(yaw_rate)
        msg.vector.y = float(vel_mean)
        msg.vector.z = float(is_r)
        self.pub.publish(msg)


class LivePredictionListenerNode(Node):
    """Subscribes to /vo/failure_prediction and records live receiver-side arrival metrics."""

    def __init__(self, prediction_topic: str = "/vo/failure_prediction"):
        super().__init__("live_prediction_listener_node")
        self.sub = self.create_subscription(PointStamped, prediction_topic, self.pred_callback, 10)
        self.received_predictions = []

    def pred_callback(self, msg: PointStamped):
        t_recv = time.perf_counter()
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.received_predictions.append({
            "recv_wall_time": t_recv,
            "telemetry_timestamp": ts,
            "prob": msg.point.x,
            "pred_label": int(msg.point.y),
            "e2e_lat_ms": msg.point.z,
        })


def main():
    repo_root = Path(__file__).resolve().parent.parent
    processed_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "models"
    data_dir = Path(os.environ.get("AEGIS_DATA_DIR", repo_root / "data" / "raw"))

    print("=================================================================")
    print("Research 2 — Live ROS 2 Deployment Validation (Condition: A_C)")
    print("=================================================================")

    # Select validated flight condition from eligible sweep pool
    test_flight = "sweep_A_C_R1"
    flight_dir = data_dir / test_flight

    if not flight_dir.exists():
        raise FileNotFoundError(f"Missing flight data directory: {flight_dir}")

    gt = pd.read_csv(flight_dir / "dataset_gt.csv")
    raw = pd.read_csv(flight_dir / "raw_vo.csv")

    active_idx = gt["pos_z"].astype(float) >= 2.0
    t0 = gt.loc[active_idx, "timestamp_total_sec"].astype(float).min()
    t1 = gt.loc[active_idx, "timestamp_total_sec"].astype(float).max()
    in_window = (raw["timestamp_total_sec"] >= t0) & (raw["timestamp_total_sec"] <= t1)
    df_active = raw.loc[in_window].copy().reset_index(drop=True)
    total_samples = len(df_active)

    print(f"Loaded {total_samples} active telemetry frames from validated condition {test_flight}.")

    # Initialize ROS 2
    rclpy.init()

    # Instantiate nodes
    pred_node = LiveFailurePredictorNode(
        scaler_path=str(models_dir / "expanded_scaler.joblib"),
        model_path=str(models_dir / "expanded_mlp.pt"),
        telemetry_topic="/telemetry/motion",
        prediction_topic="/vo/failure_prediction",
        log_csv_path=str(processed_dir / "live_ros2_predictions.csv"),
    )
    streamer_node = LiveTelemetryStreamerNode(telemetry_topic="/telemetry/motion")
    listener_node = LivePredictionListenerNode(prediction_topic="/vo/failure_prediction")

    # Multi-threaded executor to run ROS 2 pub/sub concurrently
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(pred_node)
    executor.add_node(listener_node)
    executor.add_node(streamer_node)

    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    print("\n[START] Streaming telemetry over ROS 2 topic /telemetry/motion...")
    t_start_live = time.perf_counter()

    # Stream frames with high-rate delivery (~30 Hz simulation pacing)
    inter_frame_sleep = 0.002  # Accelerated real-time streaming to test maximum throughput

    for idx, row in df_active.iterrows():
        ts = float(row["timestamp_total_sec"])
        yaw_rate = float(row["eis_yaw_rate_deg"])
        vel_mean = float(row["feature_vel_mean"])
        is_r = int(row["is_r_frame"])

        streamer_node.publish_sample(ts, yaw_rate, vel_mean, is_r)
        time.sleep(inter_frame_sleep)

        if (idx + 1) % 150 == 0:
            print(f"  Streamed {idx + 1}/{total_samples} frames...")

    # Allow buffer to drain
    time.sleep(0.5)
    t_total_live = time.perf_counter() - t_start_live
    print(f"\n[DONE] Streaming complete: {total_samples} frames in {t_total_live:.3f}s ({total_samples / t_total_live:.1f} Hz).")

    # Clean shutdown
    pred_node.destroy_node()
    listener_node.destroy_node()
    streamer_node.destroy_node()
    rclpy.shutdown()
    exec_thread.join(timeout=1.0)

    # Post-run evaluation of the live log
    live_csv = processed_dir / "live_ros2_predictions.csv"
    df_live = pd.read_csv(live_csv)
    print(f"Saved live prediction log to: {live_csv.name} ({len(df_live)} rows)")

    # Latency distribution in live ROS 2 operation
    valid_live = df_live[df_live["has_prediction"] == 1].copy()
    e2e_ms = valid_live["e2e_lat_us"] / 1000.0

    lat_live = {
        "mean_ms": float(e2e_ms.mean()),
        "median_ms": float(e2e_ms.median()),
        "p95_ms": float(np.percentile(e2e_ms, 95)),
        "p99_ms": float(np.percentile(e2e_ms, 99)),
        "max_ms": float(e2e_ms.max()),
        "total_messages": int(len(df_live)),
        "predictions_generated": int(len(valid_live)),
        "effective_rate_hz": float(total_samples / t_total_live),
    }

    # Post-flight synchronization against actual VO failure frames
    df_live["actual_inliers"] = df_active["num_inliers_pose"]
    df_live["actual_is_failure"] = (df_active["num_inliers_pose"] < 8).astype(int)

    # Contiguous failure episodes
    failure_episodes = []
    in_ep = False
    start_i = None
    for i, row in df_live.iterrows():
        if row["actual_is_failure"] == 1:
            if not in_ep:
                in_ep = True
                start_i = i
        else:
            if in_ep:
                in_ep = False
                failure_episodes.append((start_i, i - 1))
    if in_ep:
        failure_episodes.append((start_i, len(df_live) - 1))

    live_early_warns = 0
    live_post_dets = 0
    live_missed = 0
    live_lead_times = []

    for ep_start, ep_end in failure_episodes:
        t_onset = df_live.loc[ep_start, "telemetry_timestamp_s"]
        pre_win = df_live.loc[max(0, ep_start - 5): ep_start - 1]
        pre_warns = pre_win[pre_win["predicted_label"] == 1]
        during_win = df_live.loc[ep_start: ep_end]
        during_warns = during_win[during_win["predicted_label"] == 1]

        if len(pre_warns) > 0:
            live_early_warns += 1
            t_first_w = pre_warns["telemetry_timestamp_s"].min()
            lead = t_onset - t_first_w
            live_lead_times.append(lead)
        elif len(during_warns) > 0:
            live_post_dets += 1
        else:
            live_missed += 1

    summary = {
        "flight_condition": test_flight,
        "motion_cell": "A_C (Aggressive Circle, 45 deg/s)",
        "ros2_node_executed": True,
        "telemetry_topic": "/telemetry/motion",
        "prediction_topic": "/vo/failure_prediction",
        "latency_live": lat_live,
        "synchronization": {
            "total_failure_episodes": len(failure_episodes),
            "true_early_warnings": live_early_warns,
            "post_failure_detections": live_post_dets,
            "missed_failures": live_missed,
            "early_warning_rate_pct": float(live_early_warns / len(failure_episodes) * 100.0) if failure_episodes else 0.0,
            "mean_lead_time_s": float(np.mean(live_lead_times)) if live_lead_times else 0.0,
            "median_lead_time_s": float(np.median(live_lead_times)) if live_lead_times else 0.0,
        },
    }

    out_json = processed_dir / "live_ros2_validation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved live ROS 2 validation summary to: {out_json.name}")

    print("\n--- Live ROS 2 Validation Results ---")
    print(f"  Live Messages Ingested: {lat_live['total_messages']}")
    print(f"  Predictions Generated : {lat_live['predictions_generated']}")
    print(f"  Mean End-to-End Latency: {lat_live['mean_ms']:.4f} ms")
    print(f"  P99 End-to-End Latency : {lat_live['p99_ms']:.4f} ms")
    print(f"  Failure Episodes       : {len(failure_episodes)}")
    print(f"  True Early Warnings    : {live_early_warns} ({summary['synchronization']['early_warning_rate_pct']:.1f}%)")
    print(f"  Post-Failure Detections: {live_post_dets}")
    print(f"  Missed Failures        : {live_missed}")
    print(f"  Median Lead Time       : {summary['synchronization']['median_lead_time_s']:.3f} s")


if __name__ == "__main__":
    main()
