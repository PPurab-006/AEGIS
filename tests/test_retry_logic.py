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
import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
