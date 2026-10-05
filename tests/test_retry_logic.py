#!/usr/bin/env python3
"""
tests/test_retry_logic.py

Unit tests for non-destructive retry and attempt-numbering discipline
in scripts/11_run_sweep_batch.py.

Tests:
  1. Attempt 1 fails, Attempt 2 passes:
     - Directory naming: <flight>_attempt1 preserved, <flight> keeps Attempt 2 data.
     - attempts_used == 2, status == 'PASS'.
     - Data in attempt 1 is intact and never deleted or overwritten.
  2. Attempt 1 fails, Attempt 2 fails:
     - Directory naming: <flight>_attempt1 preserved, <flight> keeps final failed attempt.
     - <flight>_attempt2 does NOT exist (final failed attempt keeps plain name).
     - attempts_used == 2, status == 'FAIL'.
     - All attempt data intact.
  3. Clean flight (Attempt 1 passes):
     - Directory naming: <flight> keeps Attempt 1 data, no _attempt dirs.
     - attempts_used == 1, status == 'PASS'.
  4. preserve_attempt_directory overwrite protection:
     - Raises RuntimeError if target _attempt<N> directory already exists.
"""

import importlib
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy

# Import 11_run_sweep_batch via importlib due to leading digits
batch_module = importlib.import_module("scripts.11_run_sweep_batch")
run_flight = batch_module.run_flight
preserve_attempt_directory = batch_module.preserve_attempt_directory


def create_dummy_flight_files(flight_dir: Path, attempt_marker: str):
    """Write valid minimal dataset files plus an attempt-specific marker file."""
    flight_dir.mkdir(parents=True, exist_ok=True)
    marker_file = flight_dir / f"attempt_marker_{attempt_marker}.txt"
    marker_file.write_text(f"data_for_{attempt_marker}")

    n_samples = 120
    ts = [10.0 + i * 0.033 for i in range(n_samples)]
    df_gt = pd.DataFrame({
        "sample_idx": range(n_samples),
        "timestamp_sec": ts,
        "timestamp_nanosec": [0] * n_samples,
        "timestamp_total_sec": ts,
        "pos_x": [14.0505] * n_samples,
        "pos_y": [-7.5229] * n_samples,
        "pos_z": [2.5] * n_samples,
        "rot_x": [0.0] * n_samples,
        "rot_y": [0.0] * n_samples,
        "rot_z": [0.0] * n_samples,
        "rot_w": [1.0] * n_samples,
    })
    df_gt.to_csv(flight_dir / "dataset_gt.csv", index=False)

    df_vo = pd.DataFrame({
        "frame_idx": range(n_samples),
        "timestamp_sec": ts,
        "timestamp_nanosec": [0] * n_samples,
        "timestamp_total_sec": ts,
        "pos_x": [14.0505] * n_samples,
        "pos_y": [-7.5229] * n_samples,
        "pos_z": [2.5] * n_samples,
        "rot_x": [0.0] * n_samples,
        "rot_y": [0.0] * n_samples,
        "rot_z": [0.0] * n_samples,
        "rot_w": [1.0] * n_samples,
        "num_inliers_pose": [25] * n_samples,
        "feature_vel_mean": [4.0] * n_samples,
    })
    df_vo.to_csv(flight_dir / "raw_vo.csv", index=False)

    df_cam = pd.DataFrame({
        "frame_idx": range(n_samples),
        "timestamp_sec": ts,
        "timestamp_nanosec": [0] * n_samples,
        "timestamp_total_sec": ts,
        "filename": [f"img_{i:04d}.png" for i in range(n_samples)],
        "width": [640] * n_samples,
        "height": [480] * n_samples,
    })
    df_cam.to_csv(flight_dir / "camera_frames.csv", index=False)

    img_dir = flight_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        (img_dir / f"img_{i:04d}.png").touch()


class TestRetryLogic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.dataset_dir = self.temp_path / "datasets"
        self.raw_dir = self.temp_path / "raw"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Patch module-level paths
        self.orig_dataset_dir = batch_module.DATASET_DIR
        self.orig_raw_dir = batch_module.R2_RAW_DIR
        batch_module.DATASET_DIR = self.dataset_dir
        batch_module.R2_RAW_DIR = self.raw_dir

    def tearDown(self):
        batch_module.DATASET_DIR = self.orig_dataset_dir
        batch_module.R2_RAW_DIR = self.orig_raw_dir
        self.temp_dir.cleanup()

    def test_retry_fails_att1_passes_att2(self):
        """Variant A: Attempt 1 fails and Attempt 2 passes."""
        flight_name = "sweep_M_B_R1"
        call_count = [0]

        def stub_flight(flight_dir: Path, yaw_bin: str, motion_bin: str, env: dict):
            call_count[0] += 1
            att = call_count[0]
            if att == 1:
                create_dummy_flight_files(flight_dir, "attempt1")
                gate_res = {
                    "gate_duration": "FAIL",
                    "gate_envelope": "PASS",
                    "gate_failsafe": "PASS",
                    "gate_motion_exit": "PASS",
                    "gate_health_warn": "PASS",
                    "health_warnings_fired": "",
                    "arming_time_s": 10.85,
                    "first_tilt_exceed_45_s": "",
                    "pre_arm_notes": "",
                    "combined_pass": False,
                    "fail_reasons": ["Duration gate failed"],
                }
                return False, "Simulated Attempt 1 failure", gate_res
            else:
                create_dummy_flight_files(flight_dir, "attempt2")
                gate_res = {
                    "gate_duration": "PASS",
                    "gate_envelope": "PASS",
                    "gate_failsafe": "PASS",
                    "gate_motion_exit": "PASS",
                    "gate_health_warn": "PASS",
                    "health_warnings_fired": "",
                    "arming_time_s": 10.85,
                    "first_tilt_exceed_45_s": "",
                    "pre_arm_notes": "",
                    "combined_pass": True,
                    "fail_reasons": [],
                }
                return True, "SUCCESS", gate_res

        log_records = []
        ok = run_flight(
            flight_name=flight_name,
            yaw_bin="M",
            motion_bin="B",
            repeat=1,
            env={},
            log_records=log_records,
            execute_attempt_fn=stub_flight,
        )

        self.assertTrue(ok)
        self.assertEqual(call_count[0], 2)
        self.assertEqual(len(log_records), 1)
        rec = log_records[0]
        self.assertEqual(rec["status"], "PASS")
        self.assertEqual(rec["attempts_used"], 2)
        self.assertIn("Attempt 1 failed", rec["fail_reason"])

        # Check directories on disk
        att1_dir = self.dataset_dir / f"{flight_name}_attempt1"
        final_dir = self.dataset_dir / flight_name
        att2_dir = self.dataset_dir / f"{flight_name}_attempt2"

        self.assertTrue(att1_dir.exists(), f"Expected {att1_dir} to exist")
        self.assertTrue((att1_dir / "attempt_marker_attempt1.txt").exists(), "Attempt 1 data deleted or overwritten")
        self.assertEqual((att1_dir / "attempt_marker_attempt1.txt").read_text(), "data_for_attempt1")

        self.assertTrue(final_dir.exists(), f"Expected final dir {final_dir} to exist")
        self.assertTrue((final_dir / "attempt_marker_attempt2.txt").exists(), "Attempt 2 data missing from final dir")
        self.assertEqual((final_dir / "attempt_marker_attempt2.txt").read_text(), "data_for_attempt2")

        # Ensure no _attempt2 directory exists (final accepted attempt keeps plain name)
        self.assertFalse(att2_dir.exists(), f"{att2_dir} should not exist; final accepted attempt keeps plain name")

    def test_retry_fails_att1_fails_att2(self):
        """Variant B: Attempt 1 fails and Attempt 2 fails."""
        flight_name = "sweep_G_H_R1"
        call_count = [0]

        def stub_flight(flight_dir: Path, yaw_bin: str, motion_bin: str, env: dict):
            call_count[0] += 1
            att = call_count[0]
            create_dummy_flight_files(flight_dir, f"attempt{att}")
            gate_res = {
                "gate_duration": "FAIL",
                "gate_envelope": "FAIL",
                "gate_failsafe": "PASS",
                "gate_motion_exit": "PASS",
                "gate_health_warn": "PASS",
                "health_warnings_fired": "",
                "arming_time_s": 10.85,
                "first_tilt_exceed_45_s": 27.8,
                "pre_arm_notes": "",
                "combined_pass": False,
                "fail_reasons": [f"Simulated fail on attempt {att}"],
            }
            return False, f"Simulated fail {att}", gate_res

        log_records = []
        ok = run_flight(
            flight_name=flight_name,
            yaw_bin="G",
            motion_bin="H",
            repeat=1,
            env={},
            log_records=log_records,
            execute_attempt_fn=stub_flight,
        )

        self.assertFalse(ok)
        self.assertEqual(call_count[0], 2)
        self.assertEqual(len(log_records), 1)
        rec = log_records[0]
        self.assertEqual(rec["status"], "FAIL")
        self.assertEqual(rec["attempts_used"], 2)
        self.assertIn("Att1: Simulated fail 1", rec["fail_reason"])
        self.assertIn("Att2: Simulated fail 2", rec["fail_reason"])

        # Check directories on disk
        att1_dir = self.dataset_dir / f"{flight_name}_attempt1"
        final_dir = self.dataset_dir / flight_name
        att2_dir = self.dataset_dir / f"{flight_name}_attempt2"

        # Attempt 1 preserved
        self.assertTrue(att1_dir.exists(), f"Expected {att1_dir} to exist")
        self.assertTrue((att1_dir / "attempt_marker_attempt1.txt").exists(), "Attempt 1 data deleted or overwritten")
        self.assertEqual((att1_dir / "attempt_marker_attempt1.txt").read_text(), "data_for_attempt1")

        # Final failed attempt keeps plain directory name
        self.assertTrue(final_dir.exists(), f"Expected final dir {final_dir} to exist")
        self.assertTrue((final_dir / "attempt_marker_attempt2.txt").exists(), "Attempt 2 data missing from final dir")
        self.assertEqual((final_dir / "attempt_marker_attempt2.txt").read_text(), "data_for_attempt2")

        # <flight>_attempt2 does NOT exist
        self.assertFalse(att2_dir.exists(), f"{att2_dir} should not exist; final failed attempt keeps plain name")

    def test_clean_flight_passes_att1(self):
        """Variant C: Attempt 1 passes on first try."""
        flight_name = "sweep_G_C_R1"
        call_count = [0]

        def stub_flight(flight_dir: Path, yaw_bin: str, motion_bin: str, env: dict):
            call_count[0] += 1
            create_dummy_flight_files(flight_dir, "attempt1")
            gate_res = {
                "gate_duration": "PASS",
                "gate_envelope": "PASS",
                "gate_failsafe": "PASS",
                "gate_motion_exit": "PASS",
                "gate_health_warn": "PASS",
                "health_warnings_fired": "",
                "arming_time_s": 10.85,
                "first_tilt_exceed_45_s": "",
                "pre_arm_notes": "",
                "combined_pass": True,
                "fail_reasons": [],
            }
            return True, "SUCCESS", gate_res

        log_records = []
        ok = run_flight(
            flight_name=flight_name,
            yaw_bin="G",
            motion_bin="C",
            repeat=1,
            env={},
            log_records=log_records,
            execute_attempt_fn=stub_flight,
        )

        self.assertTrue(ok)
        self.assertEqual(call_count[0], 1)
        self.assertEqual(len(log_records), 1)
        rec = log_records[0]
        self.assertEqual(rec["status"], "PASS")
        self.assertEqual(rec["attempts_used"], 1)

        final_dir = self.dataset_dir / flight_name
        self.assertTrue(final_dir.exists())
        self.assertTrue((final_dir / "attempt_marker_attempt1.txt").exists())
        self.assertFalse((self.dataset_dir / f"{flight_name}_attempt1").exists())

    def test_preserve_attempt_directory_safety(self):
        """Preserve attempt directory must fail loudly if target already exists."""
        flight_dir = self.dataset_dir / "test_flight"
        flight_dir.mkdir(parents=True, exist_ok=True)
        (flight_dir / "file.txt").write_text("orig")

        target_dir = self.dataset_dir / "test_flight_attempt1"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "existing.txt").write_text("dont_overwrite_me")

        with self.assertRaises(RuntimeError) as ctx:
            preserve_attempt_directory(flight_dir, 1)

        self.assertIn("already exists", str(ctx.exception))
        # Ensure existing target data was NOT modified or deleted
        self.assertTrue(target_dir.exists())
        self.assertEqual((target_dir / "existing.txt").read_text(), "dont_overwrite_me")
        self.assertTrue(flight_dir.exists())

    def test_max_attempts_one_failing(self):
        """With max_attempts=1 and a failing mock:
        - exactly one attempt runs
        - plain flight dir keeps the data
        - no _attempt dir is created
        - attempts_used == 1
        - status == 'FAIL'
        """
        flight_name = "sweep_G_H_R1"
        call_count = [0]

        def stub_failing_flight(flight_dir: Path, yaw_bin: str, motion_bin: str, env: dict):
            call_count[0] += 1
            create_dummy_flight_files(flight_dir, f"attempt{call_count[0]}")
            gate_res = {
                "gate_duration": "FAIL",
                "gate_envelope": "FAIL",
                "gate_failsafe": "PASS",
                "gate_motion_exit": "PASS",
                "gate_health_warn": "PASS",
                "health_warnings_fired": "",
                "arming_time_s": 10.85,
                "first_tilt_exceed_45_s": 27.8,
                "pre_arm_notes": "",
                "combined_pass": False,
                "fail_reasons": ["Simulated failure on attempt 1"],
            }
            return False, "Simulated failure on attempt 1", gate_res

        log_records = []
        ok = run_flight(
            flight_name=flight_name,
            yaw_bin="G",
            motion_bin="H",
            repeat=1,
            env={},
            log_records=log_records,
            execute_attempt_fn=stub_failing_flight,
            max_attempts=1,
        )

        self.assertFalse(ok)
        self.assertEqual(call_count[0], 1, "Exactly one attempt must run when max_attempts=1")
        self.assertEqual(len(log_records), 1)
        rec = log_records[0]
        self.assertEqual(rec["status"], "FAIL")
        self.assertEqual(rec["attempts_used"], 1)

        final_dir = self.dataset_dir / flight_name
        self.assertTrue(final_dir.exists(), "Plain flight dir must exist and keep data")
        self.assertTrue((final_dir / "attempt_marker_attempt1.txt").exists(), "Attempt data must be preserved in plain dir")
        self.assertEqual((final_dir / "attempt_marker_attempt1.txt").read_text(), "data_for_attempt1")

        # Assert no _attempt directory was created
        attempt_dirs = list(self.dataset_dir.glob(f"{flight_name}_attempt*"))
        self.assertEqual(len(attempt_dirs), 0, f"No _attempt dir should be created, found: {attempt_dirs}")

    def test_tilt_single_sample_spike_not_counted(self):
        """(a) Single-sample spike must NOT count as a sustained tilt excursion."""
        compute_tilt_metrics = batch_module.compute_tilt_metrics
        t_arr = np.arange(10.0, 50.0, 0.02)
        tilts = np.zeros(len(t_arr))
        z_arr = np.full(len(t_arr), 2.5)

        # Single-sample spike at t = 20.0s
        spike_idx = np.argmin(np.abs(t_arr - 20.0))
        tilts[spike_idx] = 60.0

        quats = R_scipy.from_euler("x", tilts[:, None], degrees=True).as_quat()
        df_gt = pd.DataFrame({
            "sample_idx": np.arange(len(t_arr)),
            "timestamp_total_sec": t_arr,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": z_arr,
            "rot_x": quats[:, 0],
            "rot_y": quats[:, 1],
            "rot_z": quats[:, 2],
            "rot_w": quats[:, 3],
        })

        metrics = compute_tilt_metrics(df_gt, t_arm=10.0, t_land_cmd=42.0)
        self.assertEqual(metrics["first_tilt_exceed_45_s"], "", "Single-sample spike must NOT count")
        self.assertEqual(metrics["first_tilt_z_m"], "")
        self.assertEqual(metrics["first_tilt_any_s"], round(float(t_arr[spike_idx]), 3))
        self.assertEqual(metrics["max_tilt_inflight_deg"], 60.0)

    def test_tilt_sustained_excursion_counts(self):
        """(b) Sustained excursion (>=0.1s) before the landing command counts."""
        compute_tilt_metrics = batch_module.compute_tilt_metrics
        t_arr = np.arange(10.0, 50.0, 0.02)
        tilts = np.zeros(len(t_arr))
        z_arr = np.full(len(t_arr), 2.5)

        # Sustained excursion of 8 samples (0.14s >= 0.1s) starting at t = 25.0s
        start_idx = np.argmin(np.abs(t_arr - 25.0))
        for k in range(start_idx, start_idx + 8):
            tilts[k] = 60.0
        z_arr[start_idx] = 1.85

        quats = R_scipy.from_euler("x", tilts[:, None], degrees=True).as_quat()
        df_gt = pd.DataFrame({
            "sample_idx": np.arange(len(t_arr)),
            "timestamp_total_sec": t_arr,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": z_arr,
            "rot_x": quats[:, 0],
            "rot_y": quats[:, 1],
            "rot_z": quats[:, 2],
            "rot_w": quats[:, 3],
        })

        metrics = compute_tilt_metrics(df_gt, t_arm=10.0, t_land_cmd=42.0)
        self.assertEqual(metrics["first_tilt_exceed_45_s"], round(float(t_arr[start_idx]), 3))
        self.assertEqual(metrics["first_tilt_z_m"], 1.85)
        self.assertEqual(metrics["first_tilt_any_s"], round(float(t_arr[start_idx]), 3))
        self.assertEqual(metrics["max_tilt_inflight_deg"], 60.0)

    def test_tilt_touchdown_flip_after_landing_command_not_counted(self):
        """(c) Touchdown flip after the landing command must NOT count."""
        compute_tilt_metrics = batch_module.compute_tilt_metrics
        t_arr = np.arange(10.0, 50.0, 0.02)
        tilts = np.full(len(t_arr), 5.0)  # Nominal 5 deg in-flight tilt
        z_arr = np.full(len(t_arr), 2.5)

        # Touchdown flip after landing command (landing command at 42.0s, flip at 45.0s)
        td_start_idx = np.argmin(np.abs(t_arr - 45.0))
        for k in range(td_start_idx, td_start_idx + 50):
            tilts[k] = 75.0
            z_arr[k] = 0.1

        quats = R_scipy.from_euler("x", tilts[:, None], degrees=True).as_quat()
        df_gt = pd.DataFrame({
            "sample_idx": np.arange(len(t_arr)),
            "timestamp_total_sec": t_arr,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": z_arr,
            "rot_x": quats[:, 0],
            "rot_y": quats[:, 1],
            "rot_z": quats[:, 2],
            "rot_w": quats[:, 3],
        })

        metrics = compute_tilt_metrics(df_gt, t_arm=10.0, t_land_cmd=42.0)
        self.assertEqual(metrics["first_tilt_exceed_45_s"], "", "Post-landing touchdown flip must NOT count")
        self.assertEqual(metrics["first_tilt_z_m"], "")
        self.assertEqual(metrics["first_tilt_any_s"], round(float(t_arr[td_start_idx]), 3))
        self.assertEqual(metrics["max_tilt_inflight_deg"], 5.0, "In-flight max tilt must reflect pre-landing window")


if __name__ == "__main__":
    unittest.main()
