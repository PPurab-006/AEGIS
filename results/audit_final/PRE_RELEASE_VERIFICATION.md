# AEGIS Pre-Release Forensic Verification Report
> **Note on Environment Paths**: Paths cited in this report reflect the author's local evaluation environment and host workstation setup.

**Timestamp**: October 2026  
**Auditor**: Independent Verification Assistant (Antigravity Agentic QC)  
**Standard**: TRUTH > CONSISTENCY > COMPLETENESS > PRESENTATION  
**Repository**: `PPurab-006/AEGIS` (branch: `main`)  
**Workspace Root**: `<REPO_ROOT>`  

---

## 1. Summary

| Verification Task | PASS | FAIL | NOT FOUND | Total Checks | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Task 1: Re-run Audit from Raw Data** | 5 | 0 | 0 | 5 | **PASS** |
| **Task 2: Independent Recomputation from Raw Frames** | 23 | 0 | 0 | 23 | **PASS** |
| **Task 3: Setup Facts (Evidence-Based)** | 8 | 0 | 1 | 9 | **PASS w/ 1 NOT FOUND** |
| **Task 4: Release Readiness Checks** | 7 | 5 | 0 | 12 | **ACTION REQUIRED** |
| **TOTAL** | **43** | **5** | **1** | **49** | **AUDIT VERIFIED** |

*Note on Task 4 Failures*: The 5 action items represent missing standard open-source release files (`LICENSE`, `CITATION.cff`, `.zenodo.json`), unpinned requirements (`>=` instead of `==`), and untracked audit/model directories before final release tagging.

---

## 2. Task 1: Audit Pipeline Re-Run & Parity Verification

### 2.1 Pipeline Execution
All audit scripts `scripts/audit_final/01` through `07` were executed sequentially from raw flight telemetry archives into the scratch directory `results/audit_final_rerun/`. No existing files in `results/audit_final/` or `models/audit_final/` were overwritten.

### 2.2 CSV Parity Comparison (All 18 Committed CSVs)
Every CSV in `results/audit_final/` was evaluated against the corresponding rerun file in `results/audit_final_rerun/`. Maximum absolute differences across all numeric fields were evaluated against tolerance $\le 10^{-4}$.

| File Name | Rerun Rows | Committed Rows | Max Absolute Numeric Diff | Status |
|:---|:---:|:---:|:---:|:---:|
| `FINAL_NUMBER_LEDGER.csv` | 57 | 57 | `0.0000e+00` | **PASS** |
| `baseline_comparison.csv` | 8 | 8 | `0.0000e+00` | **PASS** |
| `bootstrap_paired_ci.csv` | 25 | 25 | `0.0000e+00` | **PASS** |
| `causal_replay_comparison.csv` | 14 | 14 | `0.0000e+00` | **PASS** |
| `dataset_flight_characterization.csv` | 42 | 42 | `0.0000e+00` | **PASS** |
| `event_warning_summary.csv` | 24 | 24 | `0.0000e+00` | **PASS** |
| `frozen_legacy_vs_strict.csv` | 2 | 2 | `0.0000e+00` | **PASS** |
| `generalization_folds.csv` | 24 | 24 | `0.0000e+00` | **PASS** |
| `generalization_summary.csv` | 4 | 4 | `0.0000e+00` | **PASS** |
| `label_definition_comparison.csv` | 31,615 | 31,615 | `0.0000e+00` | **PASS** |
| `leakage_statistics.csv` | 1 | 1 | `0.0000e+00` | **PASS** |
| `live_ros2_run_a_latencies.csv` | 4 | 4 | `0.0000e+00` | **PASS** |
| `live_ros2_run_b_stress.csv` | 4 | 4 | `0.0000e+00` | **PASS** |
| `matched_duty_cycle_baselines.csv` | 28 | 28 | `0.0000e+00` | **PASS** |
| `matched_duty_cycle_summary.csv` | 12 | 12 | `0.0000e+00` | **PASS** |
| `multiseed_robustness.csv` | 6 | 6 | `0.0000e+00` | **PASS** |
| `per_flight_strict_test.csv` | 14 | 14 | `0.0000e+00` | **PASS** |
| `retrained_variants_comparison.csv` | 10 | 10 | `0.0000e+00` | **PASS** |

### 2.3 Frozen Model Scoring on Strict Task
- **Data File**: `data/processed/audit_strict_frames_k5.csv.gz`
- **Model / Scaler**: `models/expanded_mlp.pt` and `models/expanded_scaler.joblib` (frozen legacy weights evaluated under strict protocol)
- **Evaluated Test Sample**: 9,758 clean non-failing frames ($F_t = 0$), 2,951 positives
- **Scoring Results**:
  - **Strict AUROC**: `0.7665` (Paper expected: `0.7665`) -> **PASS**
  - **Strict AUPRC**: `0.5977` (Paper expected: `0.5977`) -> **PASS**
  - **Strict F1 (threshold 0.50)**: `0.5836` (Paper expected: `0.5836`) -> **PASS**

### 2.4 Retrained Models Parity (Seed 42)
Models trained exclusively on the 15 training flights with hyperparameters from `scripts/audit_final/03_models_and_baselines_forensics.py`:
- **SmallMLP (Retrained Strict V0)**:
  - Test AUROC: `0.7665` (Paper expected: `0.7665`) -> **PASS**
  - Test AUPRC: `0.5982` (Paper expected: `0.5982`) -> **PASS**
  - Test F1 (threshold 0.50): `0.5758`
- **HistGradientBoostingClassifier**:
  - Test AUROC: `0.7733` (Paper expected: `0.7733`) -> **PASS**
  - Test AUPRC: `0.6101` (Paper expected: `0.6101`) -> **PASS**

### 2.5 Causal Leakage Unit Test Suite
Command: `.venv/bin/python -m pytest tests/test_causal_leakage.py -v`
- `tests/test_causal_leakage.py::test_warmup_period`: **PASSED**
- `tests/test_causal_leakage.py::test_lag_integrity`: **PASSED**
- `tests/test_causal_leakage.py::test_model_immutability`: **PASSED**
- `tests/test_causal_leakage.py::test_stage_latencies_positive`: **PASSED**
- **Result**: `4 passed in 1.75s` -> **PASS**

---

## 3. Task 2: Independent Recomputation from Raw Telemetry

Independent verification script: `scripts/audit_final/verify_independent.py`.  
This script does not import any code from `scripts/audit/` or `scripts/audit_final/`. All metrics are derived from `data/processed/telemetry_frames.csv.gz` (31,615 active frames) and raw flight logs.

| Metric / Assertion | Computed Value | Paper Expected Value | Tolerance | Status |
|:---|:---:|:---:|:---:|:---:|
| Total eligible flights | 42 | 42 | exact | **PASS** |
| Total raw frames | 51,041 | 51,041 | exact | **PASS** |
| Total active frames ($z \ge 2.0$ m) | 31,615 | 31,615 | exact | **PASS** |
| Total failure frames (`num_inliers_pose < 8`) | 1,978 | 1,978 | exact | **PASS** |
| Total failure episodes | 1,969 | 1,969 | exact | **PASS** |
| Share of single-frame episodes | 99.5937% | 99.59% | $\pm 0.01\%$ | **PASS** |
| Max failure episode length | 3 frames | 3 frames | exact | **PASS** |
| Train split active frames | 11,328 | 11,328 | exact | **PASS** |
| Train split failure frames | 676 | 676 | exact | **PASS** |
| Train split failure episodes | 676 | 676 | exact | **PASS** |
| Val split active frames | 9,736 | 9,736 | exact | **PASS** |
| Val split failure frames | 638 | 638 | exact | **PASS** |
| Val split failure episodes | 635 | 635 | exact | **PASS** |
| Test split active frames | 10,551 | 10,551 | exact | **PASS** |
| Test split failure frames | 664 | 664 | exact | **PASS** |
| Test split failure episodes | 658 | 658 | exact | **PASS** |
| Test legacy-evaluated frames | 10,411 | 10,411 | exact | **PASS** |
| Test currently failing frames ($F_t = 1$) | 653 | 653 | exact | **PASS** |
| Test strict clean frames ($10,411 - 653$) | 9,758 | 9,758 | exact | **PASS** |
| Test strict positive frames ($[t+1, t+5]$) | 2,951 | 2,951 | exact | **PASS** |
| $P(\text{failure} \mid \text{flow} < 0.5\text{ px})$ | 97.08% | 97.08% | $\pm 0.01\%$ | **PASS** |
| $P(\text{flow} < 0.5\text{ px} \mid \text{failure})$ | 97.32% | 97.32% | $\pm 0.01\%$ | **PASS** |
| Deterministic identity `is_r_frame == (yaw > 15°/s)` | 100.0% (0 mismatches) | 100.0% (0 mismatches) | exact | **PASS** |

---

## 4. Task 3: Setup Facts & Evidence

Machine-readable ledger: `results/audit_final/setup_facts.json`.

### a. Why Each of the Six Ineligible Flights Was Excluded
1. **`sweep_G_B_R1`**:
   - *Gating Rule*: Health warning gate and tilt angle gate (`max_tilt_motion_deg < 45.0`).
   - *Logged Reason*: `health_warn: FAIL; max_tilt_motion: 135.72 >= 45.0`. ULog recorded loss of control at $t=35.168$ s via yaw rate indicator (`validate_gates_on_existing.py:556, 591`).
2. **`sweep_G_H_R1`**:
   - *Gating Rule*: Active duration gate ($\ge 18.0$ s), spatial envelope gate ($X \in [-20, 20]$, $Y \in [-25, 25]$, $Z \in [0.5, 5.0]$), and tilt angle gate.
   - *Logged Reason*: `duration: FAIL (15.8s < 18s); envelope: FAIL; max_tilt_motion: 72.48 >= 45.0`. ULog recorded loss of control at $t=24.781$ s via vertical speed indicator (`validate_gates_on_existing.py:549, 579`).
3. **`sweep_G_H_R2`**:
   - *Gating Rule*: Active duration gate ($\ge 18.0$ s) and spatial envelope gate.
   - *Logged Reason*: `duration: FAIL (16.7s < 18s); envelope: FAIL`. Logged in `data/processed/sweep_flight_log.csv:12`: `"Att1: Active duration 16.7s < 18.0s threshold | Att2: Active duration 17.6s < 18.0s threshold"`.
4. **`sweep_G_H_R3`**:
   - *Gating Rule*: Active duration gate ($\ge 18.0$ s), health warning gate, and tilt angle gate.
   - *Logged Reason*: `duration: FAIL (7.5s < 18s); health_warn: FAIL; max_tilt_motion: 107.55 >= 45.0`. Logged in `data/processed/sweep_flight_log.csv:13`: `"Att1: Active duration 17.6s < 18.0s threshold | Att2: Active duration 12.6s < 18.0s threshold"`.
5. **`sweep_M_H_R2`**:
   - *Gating Rule*: Health warning gate and tilt angle gate.
   - *Logged Reason*: `health_warn: FAIL; max_tilt_motion: 45.36 >= 45.0`. ULog flagged loss of control at $t=35.923$ s via yaw rate indicator (`validate_gates_on_existing.py:554, 608`).
6. **`sweep_M_H_R3`**:
   - *Gating Rule*: Spatial envelope gate, health warning gate, and tilt angle gate.
   - *Logged Reason*: `envelope: FAIL; health_warn: FAIL; max_tilt_motion: 68.64 >= 45.0`. ULog flagged loss of control at $t=35.953$ s via yaw rate indicator (`validate_gates_on_existing.py:550, 604`).
- **Evidence**: `scripts/validate_gates_on_existing.py:543-561, 590-610`; `scripts/11_run_sweep_batch.py:761-765, 782-798, 889-902, 957-975`; `data/processed/sweep_flight_log.csv:5, 11-13, 24-25`.

### b. Camera Setup
- **Mounting Position**: `[0.12, 0.03, 0.242]` m relative to `base_link` (`<PX4_DIR>/Tools/simulation/gz/models/x500_mono_cam/model.sdf:10`).
- **Mounting Orientation**: `[0.0, 0.0, 0.0]` rad relative to `base_link` (`<PX4_DIR>/Tools/simulation/gz/models/x500_mono_cam/model.sdf:10`).
- **Pitch Angle**: Paper states "oriented at 45° pitch" (`README.md:66`, `scripts/build_full_preprint.py:216`). In the Gazebo SDF, the sensor pose is `0 0 0` (forward-facing in body frame); dynamic forward pitch during flight ranges from 1.4° to 5.5° (`scripts/fly_sweep_motion.py:181-213`).
- **Resolution**: Actual Gazebo camera sensor is **1280 × 960** (`<PX4_DIR>/Tools/simulation/gz/models/mono_cam/model.sdf:55-58`, confirmed in `camera_frames.csv:1`). Paper text states **640 × 480** (`scripts/build_full_preprint.py:216`).
- **Frame Rate**: **30 FPS** (`<update_rate>30</update_rate>` in `mono_cam/model.sdf:64`).
- **Field of View (FOV)**: **1.74 radians** ($\approx 99.69^\circ$ horizontal FOV) (`mono_cam/model.sdf:54`).
- **Image Topic**: `/world/{world}/model/x500_mono_cam_0/link/camera_link/sensor/camera/image` bridged to ROS 2 (`scripts/11_run_sweep_batch.py:1249, 1261`).

### c. Software and Environment Versions
- **PX4 Autopilot**: Paper text states **v1.14** (`scripts/build_full_preprint.py:214`). The local git repository at `<PX4_DIR>` is tagged `v1.18.0-beta1-209-g8aba32c862`.
- **Gazebo**: Paper text states **Gazebo Sim (Garden)** (`scripts/build_full_preprint.py:214`). Batch runner scripts and logs state **Gazebo Harmonic** (`scripts/11_run_sweep_batch.py:141, 1562`). Local simulation binary `gz sim --version` outputs **10.5.0**.
- **ROS 2 Distro**: Paper text states **ROS 2 Humble** (`scripts/build_full_preprint.py:410`). Batch runner script sources `source /opt/ros/lyrical/setup.bash` (`scripts/11_run_sweep_batch.py:145`), and host environment is `ROS_DISTRO=lyrical`.
- **Operating System**: Host machine runs **Ubuntu 26.04.1 LTS** (`resolute`), Linux kernel `7.0.0-34-generic`. Benchmark logs indicate timing evaluations were conducted on an Intel Core i9 / RTX 4090 host (`results/audit_final/live_ros2_verification.md:44`).
- **Python**: **3.14.4** (`.venv/bin/python --version`). README notes Python 3.11/3.14 (`README.md:9`).
- **OpenCV**: **4.10.0** (`.venv/bin/python -c "import cv2; print(cv2.__version__)"`).
- **PyTorch**: **2.14.0+cu130** (`.venv/bin/python -c "import torch; print(torch.__version__)"`).
- **scikit-learn**: **1.9.1** (`.venv/bin/python -c "import sklearn; print(sklearn.__version__)"`).

### d. Vehicle Model and World File Names
- **Vehicle Model**: `x500_mono_cam` (Holybro x500 quadrotor with monocular camera, PX4 SITL model `gz_x500_mono_cam`, entity `x500_mono_cam_0`) (`<PX4_DIR>/Tools/simulation/gz/models/x500_mono_cam/model.sdf:3`; `scripts/11_run_sweep_batch.py:58`).
- **World File**: `agriculture.world` (`<ROS_REPO>/configs/gazebo_models_worlds_collection-master/worlds/agriculture.world:1`; `scripts/11_run_sweep_batch.py:1216`; `README.md:66`).

### e. Visual Odometry Parameters in `minimal_vo.py`
Path: `<ROS_REPO>/src/core/minimal_vo.py` (cited in `results/audit_final/failure_mechanism_report.md:38-50`):
- **`goodFeaturesToTrack`**: `maxCorners=2000, qualityLevel=0.001, minDistance=5` (`minimal_vo.py:259`).
- **KLT (`calcOpticalFlowPyrLK`)**: `winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)` (`minimal_vo.py:269-270`).
- **`findEssentialMat`**: `method=cv2.RANSAC, prob=0.999, threshold=1.0` (`minimal_vo.py:302-304`).
- **Failure Threshold**: Evaluated as failure when `num_inliers_pose < 8` (or `num_matched < 8`) (`minimal_vo.py:299, 325, 327`).
- **Cache-Clear Logic**:
  - When `num_matched < 8`, pipeline executes `self.prev_pts = None` (`minimal_vo.py:348`).
  - On the immediate next frame, `if self.prev_img is None or self.prev_pts is None or len(self.prev_pts) < 100:` triggers fresh corner re-detection (`minimal_vo.py:258-259`), returning 0 matched and 0 inliers (skipping pose integration for 1 frame).
  - When `num_inliers_pose >= 8`, `self.prev_pts` retains only verified pose inliers: `self.prev_pts = pts2[inlier_mask].reshape(-1, 1, 2)` (`minimal_vo.py:342`).

### f. Yaw-Rate Commands, Motion Profiles, and Ramp Dynamics
- **Commanded Yaw Rates**:
  - Gentle (G): `7.5` deg/s (`scripts/fly_sweep_motion.py:33`).
  - Moderate (M): `20.0` deg/s (`scripts/fly_sweep_motion.py:34`). (Paper caption states 22.5 deg/s in `captions.md:15`).
  - Aggressive (A): `45.0` deg/s (`scripts/fly_sweep_motion.py:35`).
  - Extreme (E): `90.0` deg/s (`scripts/fly_sweep_motion.py:36`).
- **Motion Profiles**:
  - C (Cruise): Constant gentle forward pitch `pitch_cmd_deg = 1.4` (`scripts/fly_sweep_motion.py:181`).
  - B (Braking): Cruise pitch 1.8 deg with 4 discrete braking pulses at `(4.0, 5.5)`, `(8.5, 10.0)`, `(13.0, 14.5)`, `(17.5, 19.0)` s pulsing to `-2.0` deg for 0.4 s then holding `0.0` deg (`scripts/fly_sweep_motion.py:186-194`).
  - S (Sharp Stop): Cruise pitch 2.2 deg with 2 abrupt full stops at `(5.5, 8.5)`, `(13.5, 16.5)` s pulsing to `-3.0` deg for 0.5 s then holding `0.0` deg (`scripts/fly_sweep_motion.py:198-206`).
  - H (High-speed burst): Fast cruise pitch 3.5 deg with 2 forward surges at `(5.0, 7.5)`, `(13.0, 15.5)` s increasing pitch to `5.5` deg (`scripts/fly_sweep_motion.py:210-214`).
  *(Note: Paper describes C, B, S, H as trajectory geometries "Circle, Box, Straight line, Helix", but flight script defines them as pitch/velocity profiles).*
- **Continuous One-Direction Yaw Ramp**:
  **YES**. Implemented as `yaw_target_deg = yaw_rate_deg_s * elapsed` (`scripts/fly_sweep_motion.py:173-174`). Yaw target increases monotonically in one direction throughout the entire flight.

### g. ROS 2 Message Types and Topics (Three-Process Harness)
Implementation: `scripts/audit/live_ros2_v2.py`:
1. **Telemetry Publisher Process (`audit_publisher_node`)**:
   - Topic: `/telemetry/motion`
   - Message Type: `geometry_msgs/msg/Vector3Stamped` (`vector.x = yaw_rate`, `vector.y = optical_flow`, `vector.z = is_r_frame`) (`scripts/audit/live_ros2_v2.py:170, 186-194`).
2. **Predictor Process (`audit_predictor_node`)**:
   - Subscribes to `/telemetry/motion` (`geometry_msgs/msg/Vector3Stamped`) (`scripts/audit/live_ros2_v2.py:108`).
   - Publishes to `/vo/failure_prediction`
   - Message Type: `geometry_msgs/msg/PointStamped` (`point.x = prob`, `point.y = pred_label`, `point.z = cb_compute_ms`, `frame_id = latency string`) (`scripts/audit/live_ros2_v2.py:97-106`).
3. **Diagnostics Receiver Process (`audit_listener_node`)**:
   - Subscribes to `/vo/failure_prediction` (`geometry_msgs/msg/PointStamped`) (`scripts/audit/live_ros2_v2.py:128`).
*(Note: Preprint text at line 411 states `sensor_msgs/msg/Imu and custom telemetry messages`, but implementation uses `Vector3Stamped` and `PointStamped`).*

### h. Host Used for ROS 2 Timing Runs A & B
- **Documented Evaluation Host**: Intel Core i9 / RTX 4090, ROS 2 Humble (`results/audit_final/live_ros2_verification.md:44`; `scripts/audit_final/06_causal_replay_and_ros2_forensics.py:294`).
- **CPU Model String**: **NOT FOUND** beyond `"Intel Core i9"`.
- **Core Count**: **NOT FOUND**.
- **RAM**: **NOT FOUND**.
- **OS Build**: **NOT FOUND** beyond Linux / Ubuntu.
- **Kernel Version**: **NOT FOUND**.
*(Note: Local machine running verification is an ASUS Vivobook 14X with a 13th Gen Intel Core i5-13500H, 12 cores, 16 threads, 16 GB RAM, Ubuntu 26.04.1 LTS, kernel 7.0.0-34-generic; because logs specify that timing runs were conducted on an Intel Core i9 system, local specs do not apply).*

### i. Random Seeds Used
- **Flight Splitting**: `SEED = 42` (`data/processed/expanded_flight_split.csv:1`; `scripts/audit_final/01_dataset_and_failure_forensics.py:53`).
- **Model Training (V0 Strict)**: `SEED = 42` (`scripts/audit_final/03_models_and_baselines_forensics.py:58`).
- **HistGradientBoosting**: `random_state = 42` (`scripts/audit_final/03_models_and_baselines_forensics.py:387`).
- **LogisticRegression**: `random_state = 42` (`scripts/audit_final/03_models_and_baselines_forensics.py:408`).
- **Bootstrap Resampling (2,000 reps)**: `RandomState(42)` (`scripts/audit_final/05_uncertainty_and_generalization.py:330`).
- **Cross-Validation Split Shuffling**: `random.Random(42)` (`scripts/audit_final/05_uncertainty_and_generalization.py:507`).
- **Multi-Seed Robustness Evaluation**: `seeds = [0, 1, 2, 3, 4]` (`scripts/audit_final/05_uncertainty_and_generalization.py:456`).

---

## 5. Task 4: Release Readiness Checklist

| Item | Status | Evidence / Details |
|:---|:---:|:---|
| **Git Remote URL** | **PASS** | `origin https://github.com/PPurab-006/IMUBased_Monitor.git` |
| **GitHub Owner / Repo** | **PASS** | `PPurab-006/IMUBased_Monitor` |
| **Default Branch** | **PASS** | `main` |
| **Working Tree Clean** | **FAIL** | Modified files: `README.md`, `data/processed/expanded_generalization_report.md`, `requirements.txt`, `results/figures/publication/fig2_*`, `scripts/13-15`. Untracked files present. |
| **LICENSE file** | **FAIL / NOT FOUND** | Does not exist in repository root. |
| **CITATION.cff** | **FAIL / NOT FOUND** | Does not exist in repository root (drafted below). |
| **.zenodo.json** | **FAIL / NOT FOUND** | Does not exist in repository root (drafted below). |
| **README Reproduce Section** | **PASS** | Present at `README.md:203` (`## Reproducibility`). |
| **docs/REPRODUCIBILITY.md** | **PASS** | Exists (`docs/REPRODUCIBILITY.md`). |
| **requirements.txt Pinned Versions** | **FAIL** | Contains minimum version constraints (`>=`), e.g. `torch>=2.0`, `scikit-learn>=1.3`, not pinned exact versions (`==`). |
| **Total Repo Workspace Size** | **PASS** | `221 MB` (excluding `.venv` and `.git`). `.git` is `4.3 MB`. |
| **10 Largest Files in Repo** | **PASS** | 1. `.pytest_cache/.../nodeids` (16 MB)<br>2. `audit_strict_frames_k5.csv` (14 MB)<br>3. `audit_strict_frames_k3.csv` (14 MB)<br>4. `audit_strict_frames_k2.csv` (14 MB)<br>5. `AEGIS_Preprint_Package.zip` (9.3 MB)<br>6. `AEGIS_Audit_Verification_Bundle.zip` (9.0 MB)<br>7. `frames_with_features.csv` (7.7 MB)<br>8. `frames_labeled.csv` (7.6 MB)<br>9. `audit_strict_frames_k5.csv.gz` (4.7 MB)<br>10. `scratch/audit_strict_frames_k5.csv.gz` (4.7 MB) |
| **GitHub 100 MB Limit** | **PASS** | Zero files exceed 100 MB (largest file is 16 MB). |
| **Git LFS Tracking** | **PASS** | Zero files tracked by Git LFS. |
| **Tracking Status of Audit Assets** | **FAIL** | `results/audit_final/`, `models/audit_final/`, `telemetry_frames.csv.gz`, `audit_strict_frames_k5.csv.gz`, `docs/` are untracked (`??` in `git status`). |
| **SHA-256 Checksum: `v0_strict_seed42.pt`** | **PASS** | Actual: `bbbd9ce5020d9d46ae59da22dcf82bfae2efd09d7f98f0e482fbf24dec921f43` (Matches paper value exactly). |
| **SHA-256 Checksum: `v0_strict_scaler.joblib`** | **PASS** | Actual: `b444e21481c5e28ae5308a4b659ca5b7e6c01d82b124e0c405241fdc051282a7` (Matches paper value exactly). |
| **REPRODUCIBILITY.md Checksum Parity** | **FAIL** | `docs/REPRODUCIBILITY.md` references legacy filenames (`best_model.pth`, `retrained_v0_strict.pth`) and lacks the SHA-256 checksums of `v0_strict_seed42.pt` and `v0_strict_scaler.joblib`. |
| **Secrets / API Keys in Tracked Files** | **PASS** | Zero API keys, tokens, or credentials found. |
| **Absolute Local Paths in Tracked Files** | **PASS (0 paths remaining)** | All occurrences of absolute paths in tracked and audit files have been stripped and replaced with repo-relative paths and environment variable fallbacks (`AEGIS_DATA_DIR`, `<REPO_ROOT>`, `<RAW_DATA_DIR>`, `<PX4_DIR>`, `<ROS_REPO>`). |

---

### Suggested Release Metadata Drafts (Text Only — Not Created)

#### Suggested `CITATION.cff`
```yaml
cff-version: 1.2.0
message: "If you use this software or benchmark dataset, please cite it as below."
title: "AEGIS: Telemetry-Based Early Warning of Monocular Visual Odometry Tracking Dropouts During Aggressive UAV Flight"
authors:
  - family-names: Sen
    given-names: Purab
    orcid: "https://orcid.org/0009-0002-8385-1435"
    affiliation: "Independent Researcher"
date-released: 2026-10-09
url: "https://github.com/PPurab-006/IMUBased_Monitor"
repository-code: "https://github.com/PPurab-006/IMUBased_Monitor"
keywords:
  - visual odometry
  - early warning
  - failure prediction
  - UAV
  - telemetry
  - optical flow
  - ROS 2
```

#### Suggested `.zenodo.json`
```json
{
  "title": "AEGIS: Telemetry-Based Early Warning of Monocular Visual Odometry Tracking Dropouts During Aggressive UAV Flight",
  "creators": [
    {
      "name": "Sen, Purab",
      "orcid": "0009-0002-8385-1435",
      "affiliation": "Independent Researcher"
    }
  ],
  "description": "Replication package and benchmark artifacts for AEGIS: Learned early warning of monocular visual odometry tracking dropouts from pre-failure kinematic and optical flow telemetry during aggressive UAV flight.",
  "keywords": [
    "visual odometry",
    "failure prediction",
    "early warning",
    "UAV flight",
    "telemetry",
    "ROS 2"
  ],
  "license": "Apache-2.0",
  "upload_type": "software",
  "access_right": "open"
}
```

---

## 6. Things that Disagree with the Paper

The following 8 discrepancies between the codebase/models and the manuscript prose were conclusively identified through forensic inspection:

1. **Camera Sensor Resolution (1280×960 vs 640×480)**:  
   - *Paper Claim*: Manuscript text states "streaming 640 × 480 grayscale imagery at 30 frames per second" (`scripts/build_full_preprint.py:216`, `README.md:66`).  
   - *Codebase Reality*: The actual Gazebo camera SDF model (`mono_cam/model.sdf:55-58`) and the recorded flight telemetry logs (`camera_frames.csv:1`, `phase4_live_validation_report.md:26`) record at **1280 × 960** pixels.

2. **Camera Physical Pitch Angle (0° Body-Aligned vs 45° Downward)**:  
   - *Paper Claim*: Manuscript states "camera oriented at 45° pitch (downward/forward-facing)" (`scripts/build_full_preprint.py:216`, `README.md:66`).  
   - *Codebase Reality*: In `<PX4_DIR>/Tools/simulation/gz/models/x500_mono_cam/model.sdf:10`, the camera link pose relative to `base_link` is `<pose>.12 .03 .242 0 0 0</pose>` (0° pitch relative to body). Downward orientation during flight occurs via vehicle forward pitch tilt (1.4° to 5.5° commanded in `scripts/fly_sweep_motion.py:181-213`), not a mechanical 45° bracket mount.

3. **Moderate Commanded Yaw Rate (20.0°/s vs 22.5°/s)**:  
   - *Paper Claim*: Figure 1 caption states "Moderate 22.5°/s" (`results/figures/publication/captions.md:15`).  
   - *Codebase Reality*: The flight controller script (`scripts/fly_sweep_motion.py:34`) and flight logs (`data/processed/sweep_flight_log.csv:14`) command **20.0°/s** for bin `M`.

4. **Trajectory Bin Geometry Framing (Pitch Profiles vs Geometric Shapes)**:  
   - *Paper Claim*: Text and captions describe trajectory bins C, B, S, H as spatial geometric paths: "Circle, Box, Straight line, Helix" (`scripts/build_full_preprint.py:259`, `results/figures/publication/captions.md:15`).  
   - *Codebase Reality*: The motion generator (`scripts/fly_sweep_motion.py:176-215`) defines C, B, S, H as forward pitch dynamic profiles: "Cruise (C), Braking (B), Sharp Stop (S), and High-speed burst (H)", while yaw is commanded as a continuous unidirectional ramp (`yaw_target_deg = yaw_rate_deg_s * elapsed`). The drone does not execute closed geometrical boxes or helices in XY space.

5. **Gazebo Simulator Distribution (Harmonic vs Garden)**:  
   - *Paper Claim*: Text cites "Gazebo Sim (Garden)" (`scripts/build_full_preprint.py:214`).  
   - *Codebase Reality*: Batch runner scripts and logs explicitly configure and document "Gazebo Harmonic" (`scripts/11_run_sweep_batch.py:141, 1562`, `data/processed/sweep_batch_report.md:5`), and the host system binary `gz sim --version` is `10.5.0`.

6. **PX4 Autopilot Software Version (v1.18 vs v1.14)**:  
   - *Paper Claim*: Text states "PX4 Autopilot (v1.14)" (`scripts/build_full_preprint.py:214`).  
   - *Codebase Reality*: The PX4-Autopilot repository on the simulation host (`<PX4_DIR>`) is tagged `v1.18.0-beta1-209-g8aba32c862`.

7. **ROS 2 Middleware Distro and Topic Types**:  
   - *Paper Claim*: Text states "ROS 2 Humble deployment harness ... streaming serialized `sensor_msgs/msg/Imu` and custom telemetry messages across `/telemetry/motion`" (`scripts/build_full_preprint.py:410-411`).  
   - *Codebase Reality*: The batch script sources `ROS_DISTRO=lyrical` (`scripts/11_run_sweep_batch.py:145`), and the live multi-process harness (`scripts/audit/live_ros2_v2.py:108, 170`) exclusively uses standard `geometry_msgs/msg/Vector3Stamped` (for 3-axis telemetry ingestion) and `geometry_msgs/msg/PointStamped` (for failure predictions).

8. **`docs/REPRODUCIBILITY.md` Model Checksums**:  
   - *Paper Claim*: Paper cites verified frozen weights `v0_strict_seed42.pt` (SHA-256 `bbbd9ce5...`) and scaler `v0_strict_scaler.joblib` (SHA-256 `b444e214...`).  
   - *Documentation Reality*: `docs/REPRODUCIBILITY.md:97-103` still lists obsolete legacy file paths (`models/best_model.pth`, `models/retrained_v0_strict.pth`, `models/scaler.pkl`) and does not record the SHA-256 hashes for the strict post-audit model artifacts.
