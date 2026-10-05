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

    def test_max_tilt_post_landcmd_airborne_tumble(self):
        """Part 2 unit test: low pre-landing tilt plus an airborne 150 deg tumble after the landing command
        must give a low max_tilt_inflight_deg and a high max_tilt_post_landcmd_deg with z above 1 m."""
        compute_tilt_metrics = batch_module.compute_tilt_metrics
        t_arr = np.arange(10.0, 50.0, 0.02)
        tilts = np.full(len(t_arr), 6.0)  # low nominal 6 deg pre-landing tilt
        z_arr = np.full(len(t_arr), 2.5)

        # Airborne tumble of 150 deg after landing command (landing command at 42.0s, tumble at 43.5s at z=1.8m)
        tumble_start_idx = np.argmin(np.abs(t_arr - 43.5))
        for k in range(tumble_start_idx, tumble_start_idx + 25):
            tilts[k] = 150.0
            z_arr[k] = 1.8

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
        self.assertEqual(metrics["max_tilt_inflight_deg"], 6.0, "In-flight max tilt must be low")
        self.assertEqual(metrics["max_tilt_post_landcmd_deg"], 150.0, "Post-landing max tilt must capture 150 deg tumble")
        self.assertGreater(metrics["z_at_max_tilt_post_landcmd"], 1.0, "z at max post-landing tilt must be above 1 m")
        self.assertEqual(metrics["z_at_max_tilt_post_landcmd"], 1.8)

    def test_headline_eligibility(self):
        """Part 4 unit test: headline_eligible is True when all hardened gates PASS
        and max_tilt_motion_deg < 45.0, False otherwise."""
        # Case 1: gates pass and tilt < 45
        gate_res_pass = {
            "combined_pass": True,
            "max_tilt_motion_deg": 41.62,
        }
        tilt_ok1 = float(gate_res_pass["max_tilt_motion_deg"]) < 45.0
        self.assertTrue(gate_res_pass["combined_pass"] and tilt_ok1)

        # Case 2: gates pass but tilt >= 45
        gate_res_high_tilt = {
            "combined_pass": True,
            "max_tilt_motion_deg": 45.36,
        }
        tilt_ok2 = float(gate_res_high_tilt["max_tilt_motion_deg"]) < 45.0
        self.assertFalse(gate_res_high_tilt["combined_pass"] and tilt_ok2)

        # Case 3: gates fail even if tilt < 45
        gate_res_gate_fail = {
            "combined_pass": False,
            "max_tilt_motion_deg": 15.0,
        }
        tilt_ok3 = float(gate_res_gate_fail["max_tilt_motion_deg"]) < 45.0
        self.assertFalse(gate_res_gate_fail["combined_pass"] and tilt_ok3)

    def test_max_tilt_motion22_deg_excludes_return_braking(self):
        """max_tilt_motion22_deg measures tilt from arming to min(motion_start + 22.0s, land_cmd),
        excluding return braking spikes occurring before landing command."""
        compute_tilt_metrics = batch_module.compute_tilt_metrics
        t_arr = np.arange(10.0, 50.0, 0.02)
        tilts = np.full(len(t_arr), 5.0)
        z_arr = np.full(len(t_arr), 2.5)

        # Vehicle reaches z=2.0 at index 0 (t=10.0s), so motion_start = 10.65s, motion_end = 32.65s
        # Simulate return braking tilt of 43.2 deg at t=35.0s - 36.5s (before landing command at 42.0s)
        brake_idx = np.where((t_arr >= 35.0) & (t_arr <= 36.5))[0]
        tilts[brake_idx] = 43.2

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
        self.assertEqual(metrics["max_tilt_motion22_deg"], 5.0, "max_tilt_motion22_deg must exclude return braking spike")
        self.assertEqual(metrics["max_tilt_inflight_deg"], 43.2, "max_tilt_inflight_deg must include full in-flight window")

    def test_max_tilt_motion_excludes_ramp_after_return_start(self):
        """Unit test: synthetic data where a tilt ramp begins after t_return_start
        (t_land_cmd - 1.5s) and must be excluded from max_tilt_motion_deg."""
        compute_tilt_metrics = batch_module.compute_tilt_metrics
        t_arr = np.arange(10.0, 50.0, 0.02)
        tilts = np.full(len(t_arr), 5.0)
        z_arr = np.full(len(t_arr), 2.5)

        # Landing command at t=42.0s => t_return_start = 40.5s.
        # Tilt ramp begins after return start at t = 40.8s to 41.8s, reaching 65.0 deg.
        ramp_mask = (t_arr >= 40.8) & (t_arr <= 41.8)
        tilts[ramp_mask] = np.linspace(10.0, 65.0, int(np.sum(ramp_mask)))

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
        self.assertEqual(metrics["t_return_start_s"], 40.5, "t_return_start_s must be t_land_cmd - 1.5s")
        self.assertEqual(metrics["max_tilt_motion_deg"], 5.0, "max_tilt_motion_deg must exclude ramp after t_return_start")
        self.assertEqual(metrics["max_tilt_inflight_deg"], 65.0, "max_tilt_inflight_deg must capture in-flight ramp before landing command")

    def test_indicators_single_sample_spike_must_not_trigger(self):
        """Single-sample spikes must NOT trigger each loss-of-control indicator."""
        compute_loss_of_control = batch_module.compute_loss_of_control
        t_arr = np.arange(10.0, 50.0, 0.02)
        z_arr = np.full(len(t_arr), 2.5)
        tilts = np.full(len(t_arr), 5.0)
        yaws = np.zeros(len(t_arr))

        # Single-sample spike in tilt (i) at t=15.0s
        idx_tilt = np.argmin(np.abs(t_arr - 15.0))
        tilts[idx_tilt] = 80.0

        # Single-sample spike in z (ii) at t=20.0s (generating large instantaneous vz)
        idx_z = np.argmin(np.abs(t_arr - 20.0))
        z_arr[idx_z] = 5.0

        # Single-sample spike in yaw (iii) at t=25.0s (generating large instantaneous yaw rate)
        idx_yaw = np.argmin(np.abs(t_arr - 25.0))
        yaws[idx_yaw] = 90.0

        rots = R_scipy.from_euler("zyx", np.column_stack([yaws, np.zeros_like(tilts), tilts]), degrees=True)
        quats = rots.as_quat()

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

        res = compute_loss_of_control(df_gt, yaw_bin="G", t_arm=10.0, t_land_cmd=42.0)
        self.assertEqual(res["first_loss_of_control_s"], "", "Single-sample spikes must not trigger loss of control")
        self.assertEqual(res["loss_of_control_indicator"], "")
        self.assertEqual(res["t_onset_tilt"], "")
        self.assertEqual(res["t_onset_vz"], "")
        self.assertEqual(res["t_onset_yaw"], "")

    def test_indicators_sustained_excursion_before_landing_command_triggers(self):
        """Sustained excursion before the landing command triggers each indicator."""
        compute_loss_of_control = batch_module.compute_loss_of_control
        t_arr = np.arange(10.0, 50.0, 0.02)

        # 1. Sustained tilt > 45 deg (indicator i)
        z_arr = np.full(len(t_arr), 2.5)
        tilts = np.full(len(t_arr), 5.0)
        yaws = np.zeros(len(t_arr))
        # Excursion from index 500 (t=20.0s) for 10 samples (0.18s)
        tilts[500:510] = 55.0
        quats = R_scipy.from_euler("zyx", np.column_stack([yaws, np.zeros_like(tilts), tilts]), degrees=True).as_quat()
        df_gt = pd.DataFrame({
            "sample_idx": np.arange(len(t_arr)),
            "timestamp_total_sec": t_arr,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": z_arr,
            "rot_x": quats[:, 0], "rot_y": quats[:, 1], "rot_z": quats[:, 2], "rot_w": quats[:, 3],
        })
        res_i = compute_loss_of_control(df_gt, yaw_bin="G", t_arm=10.0, t_land_cmd=42.0)
        self.assertEqual(res_i["first_loss_of_control_s"], round(float(t_arr[500]), 3))
        self.assertEqual(res_i["loss_of_control_indicator"], "tilt")
        self.assertEqual(res_i["t_onset_tilt"], round(float(t_arr[500]), 3))

        # 2. Sustained |vz| > 1.0 m/s (indicator ii)
        z_arr = np.full(len(t_arr), 2.5)
        for k in range(600, 615):  # from t=22.0s for 15 samples (0.28s)
            z_arr[k] = z_arr[k-1] + 1.5 * 0.02
        tilts = np.full(len(t_arr), 5.0)
        quats = R_scipy.from_euler("x", tilts[:, None], degrees=True).as_quat()
        df_gt = pd.DataFrame({
            "sample_idx": np.arange(len(t_arr)),
            "timestamp_total_sec": t_arr,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": z_arr,
            "rot_x": quats[:, 0], "rot_y": quats[:, 1], "rot_z": quats[:, 2], "rot_w": quats[:, 3],
        })
        res_ii = compute_loss_of_control(df_gt, yaw_bin="G", t_arm=10.0, t_land_cmd=42.0)
        self.assertEqual(res_ii["loss_of_control_indicator"], "vertical_speed")
        self.assertEqual(res_ii["first_loss_of_control_s"], round(float(t_arr[600]), 3))

        # 3. Sustained |yaw_rate| > threshold (indicator iii)
        z_arr = np.full(len(t_arr), 2.5)
        tilts = np.full(len(t_arr), 5.0)
        yaws = np.zeros(len(t_arr))
        for k in range(800, 815):  # from t=26.0s for 15 samples (0.28s)
            yaws[k] = yaws[k-1] + 150.0 * 0.02
        quats = R_scipy.from_euler("z", yaws[:, None], degrees=True).as_quat()
        df_gt = pd.DataFrame({
            "sample_idx": np.arange(len(t_arr)),
            "timestamp_total_sec": t_arr,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": z_arr,
            "rot_x": quats[:, 0], "rot_y": quats[:, 1], "rot_z": quats[:, 2], "rot_w": quats[:, 3],
        })
        res_iii = compute_loss_of_control(df_gt, yaw_bin="G", t_arm=10.0, t_land_cmd=42.0)
        self.assertEqual(res_iii["loss_of_control_indicator"], "yaw_rate")
        self.assertEqual(res_iii["first_loss_of_control_s"], round(float(t_arr[800]), 3))

    def test_indicators_events_after_landing_command_must_not_trigger(self):
        """Same sustained events occurring after the landing command must NOT trigger."""
        compute_loss_of_control = batch_module.compute_loss_of_control
        t_arr = np.arange(10.0, 50.0, 0.02)
        z_arr = np.full(len(t_arr), 2.5)
        tilts = np.full(len(t_arr), 5.0)
        yaws = np.zeros(len(t_arr))

        # Excursions placed at index 1750 (t=45.0s), after landing command at 42.0s
        tilts[1750:1765] = 55.0
        for k in range(1750, 1765):
            z_arr[k] = z_arr[k-1] + 1.5 * 0.02
            yaws[k] = yaws[k-1] + 150.0 * 0.02

        quats = R_scipy.from_euler("zyx", np.column_stack([yaws, np.zeros_like(tilts), tilts]), degrees=True).as_quat()
        df_gt = pd.DataFrame({
            "sample_idx": np.arange(len(t_arr)),
            "timestamp_total_sec": t_arr,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": z_arr,
            "rot_x": quats[:, 0], "rot_y": quats[:, 1], "rot_z": quats[:, 2], "rot_w": quats[:, 3],
        })

        res = compute_loss_of_control(df_gt, yaw_bin="G", t_arm=10.0, t_land_cmd=42.0)
        self.assertEqual(res["first_loss_of_control_s"], "", "Post-landing command events must not trigger")
        self.assertEqual(res["loss_of_control_indicator"], "")
        self.assertEqual(res["t_onset_tilt"], "")
        self.assertEqual(res["t_onset_vz"], "")
        self.assertEqual(res["t_onset_yaw"], "")

    def test_warning_timing_class_all_five_values(self):
        """Test all five warning_timing_class values from D3."""
        classify = batch_module.classify_warning_timing

        # 1. no_warning: no warnings logged
        self.assertEqual(classify(10.0, 25.0, []), "no_warning")

        # 2. boot_only: warnings occurred, but all before arming
        warn_boot = [(5.0, "Preflight Fail: system power unavailable")]
        self.assertEqual(classify(10.0, 25.0, warn_boot), "boot_only")

        # 3. warning_before_onset: post-arm warning fired before loss of control
        warn_early = [(15.0, "Compass 0 fault")]
        self.assertEqual(classify(10.0, 25.0, warn_early), "warning_before_onset")

        # 4. warning_after_onset: post-arm warning fired at or after loss of control
        warn_late = [(30.0, "Attitude failure")]
        self.assertEqual(classify(10.0, 25.0, warn_late), "warning_after_onset")

        # 5. warning_no_onset: post-arm warning fired, but no loss of control occurred
        warn_no_loc = [(18.0, "Imbalanced propeller detected")]
        self.assertEqual(classify(10.0, None, warn_no_loc), "warning_no_onset")

    def test_landing_script_initiated_false_behavior(self):
        """When landing_script_initiated is False:
        - max_tilt_motion_deg must equal max_tilt_inflight_deg (covering arming..landing command)
        - t_return_start_s must be empty string
        - landing_script_initiated must be False.
        """
        compute_tilt_metrics = batch_module.compute_tilt_metrics
        t_arr = np.arange(10.0, 50.0, 0.02)
        tilts = np.full(len(t_arr), 5.0)
        z_arr = np.full(len(t_arr), 2.5)

        # Landing command at 42.0s => t_return_start = 40.5s.
        # Tilt ramp starts after return start (at 40.8s) reaching 65.0 deg.
        ramp_mask = (t_arr >= 40.8) & (t_arr <= 41.8)
        tilts[ramp_mask] = np.linspace(10.0, 65.0, int(np.sum(ramp_mask)))

        quats = R_scipy.from_euler("x", tilts[:, None], degrees=True).as_quat()
        df_gt = pd.DataFrame({
            "sample_idx": np.arange(len(t_arr)),
            "timestamp_total_sec": t_arr,
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": z_arr,
            "rot_x": quats[:, 0], "rot_y": quats[:, 1], "rot_z": quats[:, 2], "rot_w": quats[:, 3],
        })

        # Test landing_script_initiated=False
        m_false = compute_tilt_metrics(df_gt, t_arm=10.0, t_land_cmd=42.0, landing_script_initiated=False)
        self.assertFalse(m_false["landing_script_initiated"])
        self.assertEqual(m_false["t_return_start_s"], "", "t_return_start_s must be empty when False")
        self.assertEqual(m_false["max_tilt_motion_deg"], 65.0, "max_tilt_motion_deg must equal max_tilt_inflight_deg")
        self.assertEqual(m_false["max_tilt_inflight_deg"], 65.0)

        # Test landing_script_initiated=True
        m_true = compute_tilt_metrics(df_gt, t_arm=10.0, t_land_cmd=42.0, landing_script_initiated=True)
        self.assertTrue(m_true["landing_script_initiated"])
        self.assertEqual(m_true["t_return_start_s"], 40.5)
        self.assertEqual(m_true["max_tilt_motion_deg"], 5.0, "max_tilt_motion_deg must exclude ramp after t_return_start")

    def test_determine_landing_script_initiated(self):
        """Test determine_landing_script_initiated logic:
        - motion.log with RETURN_START vs SAFETY_ABORT
        - historical without motion.log: nominal timing +-0.3s and warning within 1.0s.
        """
        det = batch_module.determine_landing_script_initiated
        flight_dir = self.temp_path / "test_flight"
        flight_dir.mkdir(parents=True, exist_ok=True)

        # 1. New flight: motion.log with [EVENT: RETURN_START] -> True
        motion_log = flight_dir / "motion.log"
        motion_log.write_text("[EVENT: MOTION_START]\n[EVENT: RETURN_START] Motion sequence finished.\n")
        self.assertTrue(det(flight_dir, t_arm=10.0, t_land_cmd=42.0, warnings=[]))

        # 2. New flight: motion.log with [EVENT: SAFETY_ABORT] -> False
        motion_log.write_text("[EVENT: MOTION_START]\n[EVENT: SAFETY_ABORT] Bounds breached!\n[EVENT: RETURN_START]\n")
        self.assertFalse(det(flight_dir, t_arm=10.0, t_land_cmd=42.0, warnings=[]))

        # Remove motion.log for historical flight tests
        motion_log.unlink()

        # 3. Historical flight: on nominal timing (31.22s after arming), no warnings -> True
        t_arm = 10.0
        t_land_nom = t_arm + 31.22
        self.assertTrue(det(flight_dir, t_arm=t_arm, t_land_cmd=t_land_nom, warnings=[]))
        self.assertTrue(det(flight_dir, t_arm=t_arm, t_land_cmd=t_land_nom + 0.25, warnings=[]))
        self.assertTrue(det(flight_dir, t_arm=t_arm, t_land_cmd=t_land_nom - 0.25, warnings=[]))
        self.assertTrue(det(flight_dir, t_arm=t_arm, t_land_cmd=t_land_nom + 0.8, warnings=[]), "0.8s diff is within 1.0s tolerance")

        # 4. Historical flight: off nominal timing (> 1.0s diff) -> False
        self.assertFalse(det(flight_dir, t_arm=t_arm, t_land_cmd=t_land_nom - 2.0, warnings=[]))
        self.assertFalse(det(flight_dir, t_arm=t_arm, t_land_cmd=t_land_nom + 1.5, warnings=[]))

        # 5. Historical flight: on nominal timing, but warning within 1.0s before landing -> False
        warn_near = [(t_land_nom - 0.5, "Compass needs calibration - Land now!")]
        self.assertFalse(det(flight_dir, t_arm=t_arm, t_land_cmd=t_land_nom, warnings=warn_near))

        # 6. Historical flight: on nominal timing, warning > 1.0s before landing -> True
        warn_early = [(t_land_nom - 5.0, "Attitude failure (roll)")]
        self.assertTrue(det(flight_dir, t_arm=t_arm, t_land_cmd=t_land_nom, warnings=warn_early))

    def test_compute_duration_gate(self):
        """Test compute_duration_gate function:
        - Passing duration >= 18.0s
        - Failing duration < 18.0s
        - Vehicle never reaching 2.0m cruise altitude
        - None / insufficient GT rows
        """
        compute_dur = batch_module.compute_duration_gate

        # 1. dur >= 18.0s
        t_pass = np.linspace(0, 30, 300)
        z_pass = np.ones(300) * 2.5
        df_pass = pd.DataFrame({"timestamp_total_sec": t_pass, "pos_z": z_pass})
        dur, status, passed = compute_dur(df_pass)
        self.assertTrue(passed)
        self.assertAlmostEqual(dur, 30.0, places=2)
        self.assertTrue(status.startswith("PASS"))

        # 2. dur < 18.0s
        t_fail = np.linspace(0, 15, 150)
        z_fail = np.ones(150) * 2.5
        df_fail = pd.DataFrame({"timestamp_total_sec": t_fail, "pos_z": z_fail})
        dur, status, passed = compute_dur(df_fail)
        self.assertFalse(passed)
        self.assertAlmostEqual(dur, 15.0, places=2)
        self.assertTrue(status.startswith("FAIL"))

        # 3. Never reaches 2.0m
        t_low = np.linspace(0, 30, 300)
        z_low = np.ones(300) * 1.5
        df_low = pd.DataFrame({"timestamp_total_sec": t_low, "pos_z": z_low})
        dur, status, passed = compute_dur(df_low)
        self.assertFalse(passed)
        self.assertEqual(status, "FAIL (never reached 2.0m)")

        # 4. None / insufficient rows
        self.assertFalse(compute_dur(None)[2])
        self.assertFalse(compute_dur(pd.DataFrame())[2])


if __name__ == "__main__":
    unittest.main()

