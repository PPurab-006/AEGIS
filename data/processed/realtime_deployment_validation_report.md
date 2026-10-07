# Research 2 — Real-Time Deployment Validation Report

**Author / Execution**: Antigravity Autonomous Research Engine  
**Project**: Research 2 (Candidate B: Learned Telemetry-Based VO Failure Predictor)  
**Date**: October 7, 2026  
**Status**: Completed & Verified  

---

## Executive Summary

This investigation evaluates whether the trained telemetry-based Visual Odometry (VO) failure predictor can operate **causally and in real time**, relying exclusively on motion telemetry available up to the current instant, to predict imminent tracking failure before onset.

The evaluation was conducted under strict quarantine rules:
- **No model retraining, fine-tuning, or architectural adjustments**: Frozen Small MLP (`models/expanded_mlp.pt`, $15 \to 32 \to 16 \to 1$) and fitted scaler (`models/expanded_scaler.joblib`).
- **Strictly causal streaming input**: Rolling history buffer maintaining current sample ($\text{lag}0$) and historical lags ($1, 2, 3, 5$). Zero access to future telemetry, ground truth, or offline dataframes.
- **Predefined decision threshold**: Locked at $0.5$.
- **Dual validation**: Causal replay across all 14 held-out test flights (10,481 samples) followed by live ROS 2 middleware validation (`LiveFailurePredictorNode`).

---

## Section A: Causal Replay Validation

The causal replay streamed telemetry frame-by-frame in strict chronological order across the **14 held-out test flights** from the expanded 42-flight pool (covering yaw bins G, M, A, E and trajectory profiles B, C, H, S).

| Metric | Causal Streaming Replay | Offline Batch Test Baseline | Delta |
|---|---|---|---|
| **Evaluated Flights** | 14 | 14 | 0 |
| **Evaluated Samples** | 10,481 | 10,411 | +70* |
| **AUROC** | **0.8057** | **0.8046** | **+0.0011** |
| **AUPRC** | **0.7256** | **0.7260** | **-0.0004** |
| **F1 Score** | **0.6483** | **0.6484** | **-0.0001** |
| **Precision** | **0.5807** | **0.5808** | **-0.0001** |
| **Recall** | **0.7336** | **0.7336** | **0.0000** |
| **True Positives (TP)** | 2,644 | 2,644 | 0 |
| **False Positives (FP)** | 1,909 | 1,908 | +1 |
| **True Negatives (TN)** | 4,968 | 4,899 | +69* |
| **False Negatives (FN)** | 960 | 960 | 0 |

*\*Note: Causal replay evaluates frames up to the final active sample without dropping the terminal 5-frame forward window required in offline supervised dataset generation.*

### No-Future-Information Verification
- **Audit Result**: PASS (Verified via `tests/test_causal_leakage.py`).
- Feature vector construction strictly accesses `buffer[-1]` ($\text{lag}0$), `buffer[-2]` ($\text{lag}1$), `buffer[-3]` ($\text{lag}2$), `buffer[-4]` ($\text{lag}3$), and `buffer[-6]` ($\text{lag}5$).
- First 5 frames are suppressed as warm-up; zero prediction emitted until full 5-lag history is established.
- `num_inliers_pose` and ground-truth pose are strictly isolated from the live prediction path.

---

## Section B: Computational Latency & Real-Time Feasibility

Computational latency was measured separately for each execution stage across all 10,481 streaming inference cycles:

| Processing Stage | Mean ($\mu\text{s}$) | Median ($\mu\text{s}$) | P95 ($\mu\text{s}$) | P99 ($\mu\text{s}$) | Max ($\mu\text{s}$) |
|---|---|---|---|---|---|
| **Feature Construction** | 3.8 | 3.4 | 5.2 | 8.1 | 54.3 |
| **Scaler Transformation** | 80.1 | 77.2 | 102.5 | 131.0 | 794.6 |
| **MLP Forward Pass** | 43.3 | 41.6 | 58.4 | 74.1 | 8,241.0* |
| **Complete End-to-End** | **128.2** (0.128 ms) | **121.3** (0.121 ms) | **165.8** (0.166 ms) | **213.9** (0.214 ms) | **8,787.2** (8.79 ms)* |

*\*Note: Maximum latency occurred on frame 1 due to PyTorch engine / CUDA initialization; subsequent steady-state latency remained strictly below $0.25$ ms.*

### Real-Time Headroom Analysis
- **Effective Telemetry Period**: $33.333\text{ ms}$ ($30.0\text{ Hz}$).
- **Steady-State Latency Budget Used (P99)**: $0.214\text{ ms} / 33.333\text{ ms} = \mathbf{0.64\%}$.
- **Available Headroom**: **99.36%**.
- **Conclusion**: The complete inference pipeline comfortably executes in $< 1\%$ of the available sample period, proving absolute real-time feasibility.

---

## Section C: Early Warning vs. Post-Failure Detection

Across all 14 test flights, ground-truth visual odometry tracking failure events ($\text{num\_inliers\_pose} < 8$) were segmented into contiguous failure episodes. Each episode was evaluated against the streaming predictor output:

| Classification | Definition | Episodes Count | Percentage |
|---|---|---|---|
| **True Early Warning** | Prediction threshold ($\ge 0.5$) crossed *before* the first failure frame | **522** | **79.3%** |
| **Post-Failure Detection** | Threshold crossed only *after* failure onset has already occurred | **124** | **18.8%** |
| **Missed Failure** | Failure episode occurred with no warning before or during the episode | **12** | **1.8%** |
| **Total Failure Episodes** | All contiguous episodes of $\text{num\_inliers\_pose} < 8$ | **658** | **100.0%** |

### Early Warning Lead Time Statistics ($t_{\text{first\_failure}} - t_{\text{prediction}}$)
- **Median Lead Time**: **0.165 s** (~5 frames)
- **Mean Lead Time**: **0.153 s**
- **25th Percentile (P25)**: **0.165 s**
- **75th Percentile (P75)**: **0.165 s**
- **Minimum Lead Time**: **0.033 s** (1 frame prior to onset)
- **Maximum Lead Time**: **0.165 s** (5 frames prior to onset)

---

## Section D: Warning Behavior & Operational Stability

- **Total Warning Episodes**: 1,215 contiguous prediction episodes.
- **Positive State Transitions ($0 \to 1$)**: 1,215 transitions.
- **Median Warning Duration**: 0.099 s (3 frames).
- **Mean Warning Duration**: 0.151 s.
- **False-Warning Episodes**: 459 episodes occurred during intervals where no failure occurred within the $K=5$ forward window.
- **Operational Utility**: The model provides a decisive, low-latency burst of warning frames (lead time 0.165 s) before 79.3% of tracking failures, providing sufficient lead time for an onboard supervisor to freeze VO pose integration or switch to an IMU dead-reckoning fallback.

---

## Section E: Live ROS 2 Middleware Validation

The causal streaming predictor was deployed in a native ROS 2 node (`LiveFailurePredictorNode`) communicating over standard ROS 2 DDS middleware (`rclpy`).

- **Tested Flight Condition**: `sweep_A_C_R1` (Aggressive Circle, commanded yaw rate $45^\circ/\text{s}$, smooth trajectory, non-destructive, high failure rate).
- **Telemetry Ingestion Topic**: `/telemetry/motion` (`geometry_msgs/msg/Vector3Stamped`).
- **Prediction Publishing Topic**: `/vo/failure_prediction` (`geometry_msgs/msg/PointStamped`).
- **Total Live Telemetry Ingested**: 759 messages.
- **Total Live Predictions Emitted**: 754 messages (first 5 warm-up frames suppressed).
- **Live Middleware Latency**:
  - Mean: **0.6335 ms**
  - Median: **0.5120 ms**
  - P95: **1.2140 ms**
  - P99: **3.7128 ms**
  - Headroom within 33.3 ms period: **> 88.8%**.
- **Live Failure Episode Synchronization**:
  - Total Ground-Truth Failure Episodes: 76
  - **True Early Warnings in ROS 2**: **71 episodes (93.4%)**
  - Post-Failure Detections: 4 episodes (5.3%)
  - Missed Failures: 1 episode (1.3%)
  - **Median Live Lead Time**: **0.165 s**

---

## Section F: Claim Classification

Based on the verified experimental results and the strict decision criteria established in Part 13:

1. **Causal operation verified**: YES (strict rolling history, verified zero future data access, automated tests pass).
2. **End-to-end latency fits telemetry period**: YES ($0.128\text{ ms}$ steady-state, $99.36\%$ headroom in pure Python; $0.633\text{ ms}$ in live ROS 2).
3. **Predictions occur before failure**: YES (mean lead time $0.153\text{ s}$, median $0.165\text{ s}$).
4. **Meaningful fraction of failures receive early warnings**: YES ($79.3\%$ across all 14 test flights; $93.4\%$ in live ROS 2).
5. **Live ROS 2 operation demonstrated**: YES (`LiveFailurePredictorNode` ran with active pub/sub and verified synchronization).

### Final Classification
**`REAL-TIME PREDICTIVE VALIDATION SUCCESS`**

---

## Section G: Scientific Claim

> **"The proposed lightweight telemetry-based model was demonstrated to operate causally in real time and predict imminent VO failure from onboard-available motion telemetry prior to failure onset."**

*Scope Boundaries*:
- The result validates **causal, online failure prediction prior to onset**.
- It does **not** claim autonomous recovery, flight safety guarantees, or physical causal mechanisms.
- All required inputs (`eis_yaw_rate_deg`, `feature_vel_mean`, `is_r_frame`) are motion telemetry variables available directly onboard; hence the system is formally characterized as an **onboard telemetry-based imminent failure predictor**.
