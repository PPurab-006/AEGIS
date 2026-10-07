#!/usr/bin/env python3
"""
Research 2 — Live ROS 2 Node for Imminent VO Failure Prediction.

Subscribes strictly to live motion telemetry topics from the running vehicle/simulation,
maintains a rolling history buffer, constructs 15 features causally, applies the frozen
expanded scaler, evaluates the expanded Small MLP, and publishes real-time failure warnings.

CRITICAL CONSTRAINTS:
  - Receives ONLY motion telemetry:
      * eis_yaw_rate_deg (from IMU/telemetry)
      * feature_vel_mean (from visual motion tracking)
      * is_r_frame (yaw rate threshold flag)
  - FORBIDDEN in the live node:
      * VO output
      * Ground truth
      * Future trajectory information
      * Offline CSV files
  - Records timestamps for:
      * Telemetry arrival
      * Feature construction
      * Model inference
      * Prediction output
"""

import csv
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Vector3Stamped, PointStamped
from std_msgs.msg import Float32MultiArray

from scripts.streaming_failure_predictor import StreamingFailurePredictor


class LiveFailurePredictorNode(Node):
    """ROS 2 Node executing live streaming failure prediction."""

    def __init__(
        self,
        scaler_path: str = "models/expanded_scaler.joblib",
        model_path: str = "models/expanded_mlp.pt",
        telemetry_topic: str = "/telemetry/motion",
        prediction_topic: str = "/vo/failure_prediction",
        log_csv_path: str = "data/processed/live_ros2_predictions.csv",
    ):
        super().__init__("live_failure_predictor_node")

        self.declare_parameter("scaler_path", scaler_path)
        self.declare_parameter("model_path", model_path)
        self.declare_parameter("telemetry_topic", telemetry_topic)
        self.declare_parameter("prediction_topic", prediction_topic)
        self.declare_parameter("log_csv_path", log_csv_path)

        s_path = Path(self.get_parameter("scaler_path").get_parameter_value().string_value)
        m_path = Path(self.get_parameter("model_path").get_parameter_value().string_value)
        self.log_path = Path(self.get_parameter("log_csv_path").get_parameter_value().string_value)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f"Loading frozen scaler from: {s_path}")
        self.get_logger().info(f"Loading frozen MLP from: {m_path}")
        self.predictor = StreamingFailurePredictor(scaler_path=s_path, model_path=m_path, threshold=0.5)

        # QoS profile for high-rate sensor telemetry
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        t_topic = self.get_parameter("telemetry_topic").get_parameter_value().string_value
        p_topic = self.get_parameter("prediction_topic").get_parameter_value().string_value

        # Support both Vector3Stamped and Float32MultiArray
        self.sub_vec = self.create_subscription(Vector3Stamped, t_topic, self.telemetry_vector_callback, qos)
        self.sub_arr = self.create_subscription(Float32MultiArray, t_topic + "_arr", self.telemetry_array_callback, qos)

        self.pred_pub = self.create_publisher(PointStamped, p_topic, 10)

        # CSV Logging
        self.log_records = []
        self.csv_file = open(self.log_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "arrival_timestamp_s",
            "telemetry_timestamp_s",
            "has_prediction",
            "predicted_prob",
            "predicted_label",
            "feat_lat_us",
            "scale_lat_us",
            "mlp_lat_us",
            "e2e_lat_us",
            "eis_yaw_rate_deg",
            "feature_vel_mean",
            "is_r_frame",
        ])

        self.msg_count = 0
        self.pred_count = 0
        self.warn_count = 0
        self.last_pred_label = 0

        self.get_logger().info(f"Live Failure Predictor Node initialized. Subscribed to {t_topic}, publishing to {p_topic}")

    def telemetry_vector_callback(self, msg: Vector3Stamped):
        t_arrival = time.perf_counter()
        ts = float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        yaw_rate = float(msg.vector.x)
        vel_mean = float(msg.vector.y)
        is_r = int(msg.vector.z)

        self._process_telemetry(ts, yaw_rate, vel_mean, is_r, t_arrival)

    def telemetry_array_callback(self, msg: Float32MultiArray):
        t_arrival = time.perf_counter()
        if len(msg.data) >= 4:
            ts = float(msg.data[0])
            yaw_rate = float(msg.data[1])
            vel_mean = float(msg.data[2])
            is_r = int(msg.data[3])
            self._process_telemetry(ts, yaw_rate, vel_mean, is_r, t_arrival)

    def _process_telemetry(self, ts: float, yaw_rate: float, vel_mean: float, is_r: int, t_arrival: float):
        self.msg_count += 1
        telemetry = {
            "timestamp": ts,
            "eis_yaw_rate_deg": yaw_rate,
            "feature_vel_mean": vel_mean,
            "is_r_frame": is_r,
        }

        res = self.predictor.update(telemetry)
        t_output = time.perf_counter()

        has_p = res["has_prediction"]
        prob = res["prob"]
        pred_label = res["pred_label"]
        lats = res["latencies_us"]

        # Publish PointStamped prediction: x=prob, y=label, z=e2e_lat_ms
        if has_p:
            self.pred_count += 1
            out_msg = PointStamped()
            out_msg.header.stamp.sec = int(ts)
            out_msg.header.stamp.nanosec = int((ts - int(ts)) * 1e9)
            out_msg.point.x = prob
            out_msg.point.y = float(pred_label)
            out_msg.point.z = lats["end_to_end_us"] / 1000.0
            self.pred_pub.publish(out_msg)

            # Warning state transition tracking
            if pred_label == 1 and self.last_pred_label == 0:
                self.warn_count += 1
                self.get_logger().warning(f"[IMMINENT VO FAILURE WARNING] t={ts:.3f}s | Prob={prob:.4f} | Latency={lats['end_to_end_us']/1000.0:.3f}ms")
            self.last_pred_label = pred_label

        # Write to log
        self.csv_writer.writerow([
            f"{t_arrival:.6f}",
            f"{ts:.6f}",
            int(has_p),
            f"{prob:.6f}" if has_p else "",
            pred_label,
            f"{lats['feature_construction_us']:.2f}",
            f"{lats['scaler_transform_us']:.2f}",
            f"{lats['mlp_inference_us']:.2f}",
            f"{lats['end_to_end_us']:.2f}",
            f"{yaw_rate:.4f}",
            f"{vel_mean:.4f}",
            is_r,
        ])
        if self.msg_count % 100 == 0:
            self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
            self.get_logger().info(f"Flushed and closed {self.log_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LiveFailurePredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
