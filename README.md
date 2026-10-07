# Research 2 — Learned VO Failure Predictor (Archival Record)

**Project Status**: **FROZEN / COMPLETE**  
**Final Scientific Classification**: `REAL-TIME PREDICTIVE VALIDATION SUCCESS`  
**Frozen Model Checkpoint**: [`models/expanded_mlp.pt`](file:///home/purab/Purab/Projects/Research2/models/expanded_mlp.pt)  
**Frozen Scaler**: [`models/expanded_scaler.joblib`](file:///home/purab/Purab/Projects/Research2/models/expanded_scaler.joblib)  
**Primary Report**: [`data/processed/realtime_deployment_validation_report.md`](file:///home/purab/Purab/Projects/Research2/data/processed/realtime_deployment_validation_report.md)  

---

## 1. Executive Summary

Research 2 investigated whether visual odometry (VO) tracking failure on autonomous multicopters can be predicted from onboard-available motion telemetry *prior to failure onset*. 

While Research 1 established a linear correlation null result (single-variable correlation decaying monotonically with lag, all $|r| < 0.16$), Research 2 tested and confirmed that a compact, learned nonlinear model over a short backward history ($\le 165$ ms) reliably captures imminent failure precursors across held-out flight trajectories and operates comfortably within the vehicle's real-time telemetry period.

---

## 2. Research Questions & Hypotheses

1. **Prediction Feasibility**: Can a learned nonlinear model extract predictive signal of imminent VO failure within a forward horizon ($K=5$ frames, $\approx 165$ ms) using only lightweight motion telemetry?
2. **Distributional Generalization**: Does the learned predictor generalize across an expanded, aggressive flight sweep covering diverse yaw rates ($7.5^\circ/\text{s}$ to $90.0^\circ/\text{s}$) and trajectory profiles (circle, box, helix, straight), rather than merely fitting the initial 33 flights?
3. **Causal Real-Time Deployability**: Can the predictor operate strictly causally without future information, achieve end-to-end inference latency well beneath the $33.3$ ms telemetry period, and warn prior to failure onset in live ROS 2 middleware execution?

---

## 3. Dataset, Filtering, and Flight Splits

- **Original Baseline Pool**: 33 flights (24 core + 9 exploratory, F1–F11, 21,618 frames), split flight-wise into 11 train, 11 validation, 11 test.
- **Expanded Sweep Pool**: 48 flight maneuvers across a $4 \times 4$ matrix of yaw bins (G, M, A, E) and motion bins (C, B, S, H).
- **Hardened Eligibility Filtering**:
  - Excluded 6 ineligible flights: `sweep_G_B_R1`, `sweep_G_H_R1`, `sweep_G_H_R2`, `sweep_G_H_R3`, `sweep_M_H_R2`, `sweep_M_H_R3`.
  - Condition **G_H** is officially excluded from the training pool after $0/3$ fresh re-flight attempts met predefined safety and envelope gates.
  - Final eligible pool: **exactly 42 flights** (31,195 frames, 10,803 positive labels, 34.63% positive rate).
- **Flight-Level Stratified Split (Seed 42)**:
  - **Train**: 15 flights (11,178 frames, 3,739 positives, 33.45% positive rate)
  - **Validation**: 13 flights (9,606 frames, 3,460 positives, 36.02% positive rate)
  - **Held-Out Test**: 14 flights (10,411 frames, 3,604 positives, 34.62% positive rate)
  - **Zero Flight Leakage**: Flights in Train, Val, and Test are completely disjoint ($\text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$).

---

## 4. Model Architecture & Frozen Features

- **Architecture**: Small MLP ($15 \to 32 \to 16 \to 1$) with ReLU activations and sigmoid output.
- **Objective**: `BCEWithLogitsLoss` with training-derived positive weight ($pos\_weight = 1.9896$).
- **Optimizer**: Adam ($\text{lr} = 10^{-3}, \text{weight\_decay} = 10^{-4}, \text{batch\_size} = 64, \text{max\_epochs} = 30$, early stopping patience 7).
- **Feature Scaling**: `StandardScaler` fit strictly on the 15 training flights; test data scaled using frozen parameters.
- **Feature Set (15 Lagged Telemetry Features)**:
  - Absolute body yaw rate: `eis_yaw_rate_deg_lag{0, 1, 2, 3, 5}`
  - Mean optical flow magnitude: `feature_vel_mean_lag{0, 1, 2, 3, 5}`
  - Binary yaw-rate gate flag: `is_r_frame_lag{0, 1, 2, 3, 5}`
- **Label Definition**: Raw VO failure rule $\text{num\_inliers\_pose} < 8$. Target is binary indicator of failure within current or next $K=5$ frames ($\approx 165$ ms).

---

## 5. Key Experimental Findings

### 5.1 Offline Generalization Evaluation
| Metric | Original 33-Flight Baseline | Expanded 42-Flight Sweep | Generalization Verdict |
|---|---|---|---|
| **Eligible Flights** | 33 | 42 | +9 flights (+27.3%) |
| **Held-Out Test Flights** | 11 | 14 | Disjoint distribution |
| **Test AUROC ($K=5$)** | **0.8234** | **0.8046** | Delta = -0.0188 |
| **Test F1 Score ($K=5$)** | **0.6634** | **0.6484** | Delta = -0.0150 |
| **Precision** | 0.6033 | 0.5808 | Stable |
| **Recall** | 0.7368 | 0.7336 | Delta = -0.0032 |
| **AUPRC** | N/A | **0.7260** | Substantially above base rate (0.346) |

Across all 14 held-out test flights, individual per-flight AUROC ranged from **0.6690 to 0.8393** (mean ~0.760), with zero flights collapsing toward chance.

### 5.2 Horizon Sensitivity ($K \in \{2, 3, 5\}$)
- **$K=5$ (~165 ms, Primary)**: AUROC = 0.8046, F1 = 0.6484
- **$K=3$ (~99 ms)**: AUROC = 0.8052, F1 = 0.5512
- **$K=2$ (~66 ms)**: AUROC = 0.8174, F1 = 0.4789
AUROC remains virtually constant across horizons, while F1 increases monotonically with window length as positive frames accumulate.

### 5.3 Causal Streaming Replay (10,481 Samples)
- **Integrity**: Evaluated using a rolling history buffer ($N=6$). AUROC = 0.8057, F1 = 0.6483 (matches offline test identically).
- **Latency**:
  - Feature Construction: Mean = $3.8\,\mu\text{s}$
  - Scaler Transform: Mean = $80.1\,\mu\text{s}$
  - MLP Forward Pass: Mean = $43.3\,\mu\text{s}$
  - **End-to-End Latency**: Mean = **0.1282 ms** (Median 0.1213 ms, P99 0.2139 ms).
  - **Telemetry Budget Headroom**: **99.36%** of the $33.3$ ms budget remains available.
- **Lead Time**:
  - Total Failure Episodes: 658
  - **True Early Warnings**: **522 episodes (79.3%)** warned *prior to* failure onset.
  - Post-Failure Detections: 124 episodes (18.8%).
  - Missed Failures: 12 episodes (1.8%).
  - **Median Lead Time**: **0.165 s** (Mean 0.153 s, Range 0.033 s to 0.165 s).

### 5.4 Live ROS 2 Deployment Validation
- **Middleware**: ROS 2 (`rclpy`) node [`LiveFailurePredictorNode`](file:///home/purab/Purab/Projects/Research2/scripts/live_ros2_failure_predictor.py) subscribing to `/telemetry/motion` and publishing to `/vo/failure_prediction`.
- **Condition**: `sweep_A_C_R1` (Aggressive circle at $45^\circ/\text{s}$ yaw rate).
- **Inference Rate**: Sustained streaming up to $192$ Hz (far exceeding 30 Hz requirement).
- **Middleware Latency**: Mean = **0.6335 ms**, P99 = **3.7128 ms** (>88% headroom).
- **Live Early Warning Rate**: **71 / 76 failure episodes (93.4%)** received valid early warnings before failure onset, with median lead time of **0.165 s**.

---

## 6. Limitations

1. **Simulation Environment**: Data collected in PX4 SITL + Gazebo (`agriculture.world`). While aerodynamics, sensor noise, and dynamics are simulated with high fidelity, physical hardware transfer remains unvalidated.
2. **Optical Flow Conflation**: In raw (un-derotated) mode, KLT flow conflates rotational and translational visual motion.
3. **No Closed-Loop Intervention**: Research 2 evaluated predictive failure detection only; autonomous supervisor recovery or control interventions were not executed.

---

## 7. Authoritative Scientific Claim

> **"The proposed lightweight telemetry-based model was demonstrated to operate causally in real time and predict imminent VO failure from onboard-available motion telemetry prior to failure onset."**

*Boundaries*:
- This study demonstrates **causal, online failure prediction prior to onset**.
- It does **not** claim autonomous recovery, flight safety guarantees, or physical causal mechanisms.
- All required inputs (`eis_yaw_rate_deg`, `feature_vel_mean`, `is_r_frame`) are motion telemetry variables available directly onboard; hence the system is formally characterized as an **onboard telemetry-based imminent failure predictor**.

---

## 8. Archival Freeze & Transition to Research 3

- **Research 2 is FROZEN and COMPLETE.** No further training, threshold tuning, or flight experiments will be conducted in this repository.
- **Physical Hardware Testing**: Explicitly categorized as future work.
- **Research 3 Transition**: Research 3 will consume the frozen Research 2 predictor ([`models/expanded_mlp.pt`](file:///home/purab/Purab/Projects/Research2/models/expanded_mlp.pt), [`models/expanded_scaler.joblib`](file:///home/purab/Purab/Projects/Research2/models/expanded_scaler.joblib)) as an immutable failure-warning oracle to develop and evaluate autonomous supervisory recovery controllers.
