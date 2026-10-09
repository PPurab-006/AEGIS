# AEGIS: Telemetry-Based Early Warning of Monocular Visual Odometry Tracking Dropouts During Aggressive UAV Flight

A causal, lightweight neural network that anticipates imminent monocular Visual Odometry (VO) tracking collapse 165 ms ahead of time using only onboard-available vehicle motion telemetry.

| Metadata | Details |
|---|---|
| **Status** | Complete & Frozen (v1.1.0 Archival Release) |
| **Research Area** | Autonomous UAV Navigation · Visual Odometry · Real-Time Failure Anticipation |
| **Core Stack** | Python 3.14.4 · PyTorch · Scikit-Learn · ROS 2 · PX4 SITL (v1.18.0-beta1) · Gazebo Sim (gz sim 10.5.0) |
| **Dataset** | Expanded 42-Flight Parametric Sweep (31,615 active frames, 14 held-out test flights) |
| **Model Size** | 2-Hidden-Layer MLP (1,057 parameters, 0.128 ms CPU inference latency) |

---

## Overview

Monocular Visual Odometry (VO) algorithms (such as the classical 5-point Essential matrix estimator with RANSAC) suffer catastrophic tracking failures when unmanned aerial vehicles (UAVs) undergo aggressive rotational or translational maneuvers. Prior work established a single-variable linear correlation null result: instantaneous and single-lag telemetry correlations with VO inlier drops decay monotonically and remain weak ($|r| < 0.16$), ruling out simple static scalar gating thresholds.

AEGIS proves that **a compact, nonlinear model over a multi-lag temporal history ($\le 165$ ms) reliably captures imminent failure precursors across held-out flight trajectories**. Operating strictly causally on lightweight flight telemetry, the system achieves **0.8046 AUROC** on legacy protocol evaluations and **0.7665 AUROC** on strict future-only evaluations on completely unseen flight maneuvers, providing advance warnings on **79.3%** of failure episodes with a median lead time of **0.165 s**, consuming less than **1%** of the vehicle's real-time 30 Hz telemetry period.

---

## Research Questions & Core Problem

1. **Prediction Feasibility**: Can a lightweight neural model extract a reliable predictive signal of imminent VO failure within a forward horizon ($K=5$ frames, $\approx 165$ ms) using only onboard motion telemetry, bypassing expensive raw-image processing?
2. **Distributional Generalization**: Does the learned predictor generalize across an expanded, aggressive flight sweep spanning diverse yaw rates ($7.5^\circ/\text{s}$ to $90.0^\circ/\text{s}$) and motion profiles (Cruise, Braking, Sharp stop, High-speed burst), rather than overfitting a narrow flight regime?
3. **Causal Real-Time Deployability**: Can the predictor run strictly causally without future information, achieve end-to-end inference latency well within the $33.3$ ms telemetry period, and warn prior to failure onset in live ROS 2 middleware execution?

---

## System Architecture & Pipeline

The system consumes streaming motion telemetry at 30 Hz, maintains a lightweight rolling FIFO buffer ($N=6$), constructs a 15-dimensional lagged feature vector, applies a frozen preprocessor, and evaluates an MLP to emit an early failure probability $P(\text{fail}_{t+5})$.

```
Onboard Flight Telemetry (30 Hz)
 ├─ eis_yaw_rate_deg     (Body yaw rate from IMU / EIS)
 ├─ feature_vel_mean     (Mean 2D optical flow velocity from KLT tracker)
 └─ is_r_frame           (Binary keyframe / high-rate rotation indicator)
           │
           ▼
Rolling Temporal History Buffer (N=6 steps, span = 165 ms)
           │
           ▼
15-Dimensional Lagged Feature Vector
  x_t = [yaw_lag{0,1,2,3,5}, flow_lag{0,1,2,3,5}, r_flag_lag{0,1,2,3,5}]
           │
           ▼
Frozen StandardScaler (Fitted exclusively on 15 training flights)
           │
           ▼
Compact Multi-Layer Perceptron (SmallMLP: 15 → 32 → 16 → 1)
           │
           ▼
Sigmoid Failure Probability: P(fail_{t+5} | x_{t-5:t})
           │
           ▼
Decision Threshold (θ = 0.50) ──► Real-Time Warning Alert / ROS 2 Topic
```

---

## Dataset & Experimental Setup

### Simulation Environment
Data was acquired in high-fidelity PX4 Software-In-The-Loop (SITL, PX4 tree `v1.18.0-beta1`) coupled with Gazebo Sim (`gz sim 10.5.0`) in a realistic agricultural terrain (`agriculture.world`). The simulated Holybro x500 quadrotor was equipped with a body-aligned forward-facing monocular camera (pose `0 0 0` relative to `base_link`) streaming $1280 \times 960$ grayscale frames at 30 FPS with a horizontal field of view of 1.74 rad, tightly synchronized with vehicle odometry, IMU, and KLT sparse optical flow telemetry.

### Flight Sweep Design
The dataset spans a parametric $4 \times 4$ matrix combining 4 yaw rate bins and 4 motion profiles across 3 independent repeats:
- **Yaw Rate Bins**: Gentle (`G`, $7.5^\circ/\text{s}$), Moderate (`M`, $20.0^\circ/\text{s}$), Aggressive (`A`, $45.0^\circ/\text{s}$), Extreme (`E`, $90.0^\circ/\text{s}$).
- **Motion Profiles**: Cruise (`C`), Braking (`B`), Sharp stop (`S`), High-speed burst (`H`) (forward pitch dynamic profiles under continuous one-direction yaw ramp).

### Quality Control & Flight Eligibility Filtering
A strict multi-point physical flight envelope gate excluded unviable trials:
- Excluded 6 ineligible flights: `sweep_G_B_R1`, `sweep_G_H_R1`, `sweep_G_H_R2`, `sweep_G_H_R3`, `sweep_M_H_R2`, `sweep_M_H_R3`.
- The `G_H` condition was removed after $0/3$ attempts satisfied safe altitude/trajectory constraints.
- **Final Eligible Dataset**: Exactly **42 flights**, totaling **31,615 active frames** ($pos_z \ge 2.0$ m).

### Leak-Free Flight-Level Partitioning
To guarantee zero optimization leakage, all data splits were partitioned strictly at the flight level (Seed 42):
- **Train Split**: 15 flights (11,328 active frames, 11,178 processed frames)
- **Validation Split**: 13 flights (9,736 active frames, 9,606 processed frames)
- **Held-Out Test Split**: 14 flights (10,551 active frames, 10,411 processed frames)
- $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (Zero frame leakage).

---

## Methodology

### Ground Truth VO Failure Definition
Visual Odometry failure is defined by the classical 5-point Essential matrix solver health rule:
$$\text{Failure}_t = \mathbb{I}(\text{num\_inliers\_pose}_t < 8)$$
The prediction target is causal early failure anticipation at a forward horizon of $K=5$ frames ($\approx 165$ ms at 30 FPS):
$$y_t^{(K=5)} = \mathbb{I}\left(\bigvee_{k=0}^{5} \text{Failure}_{t+k} = 1\right)$$

### Temporal Feature Construction
The model relies on 15 features constructed from 3 core telemetry signals at backward lag indices $\{0, 1, 2, 3, 5\}$:
1. `eis_yaw_rate_deg_lag{0,1,2,3,5}`: Absolute body yaw rate ($^\circ/\text{s}$).
2. `feature_vel_mean_lag{0,1,2,3,5}`: Mean magnitude of 2D optical flow displacement between consecutive frames (px).
3. `is_r_frame_lag{0,1,2,3,5}`: Binary indicator signaling high angular velocity or keyframe transition.

### Model Architecture & Training
- **Architecture**: Multi-Layer Perceptron (`SmallMLP`):
  - Layer 1: Linear(15, 32) + ReLU
  - Layer 2: Linear(32, 16) + ReLU
  - Layer 3: Linear(16, 1)
  - Total Parameters: **1,057**
- **Loss Function**: `BCEWithLogitsLoss` with positive class weight $w_{\text{pos}} = 1.9896$ derived strictly from the 15 training flights.
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$, batch size 64, max 30 epochs, early stopping patience 7).

---

## Experiments & Results

### 1. Generalization Across Flight Sweeps
Comparing the original 33-flight baseline against the expanded 42-flight parametric sweep on held-out test flights:

| Experiment / Metric | Original 33-Flight Baseline | Expanded 42-Flight Sweep (Legacy Protocol) | Strict Evaluation |
|---|---|---|---|
| **Eligible Flights** | 33 | **42** | **42** |
| **Held-Out Test Flights** | 11 | **14** | **14** |
| **Total Test Frames** | 7,248 | **10,411** | **9,758** |
| **Test AUROC ($K=5$)** | **0.8234** | **0.8046** | **0.7665** |
| **Test AUPRC ($K=5$)** | N/A | **0.7260** | **0.5982** |
| **Test F1 Score** | 0.6634 | **0.6484** | **0.5775** |
| **Test Recall** | 0.7368 | **0.7336** | **0.6682** |
| **Test Precision** | 0.6033 | **0.5808** | **0.5085** |

### 2. Horizon Sensitivity Analysis ($K \in \{2, 3, 5\}$)
Evaluating the model across shorter forward prediction windows on the held-out test partition:

| Forward Horizon | Time Horizon ($\tau$) | AUROC | AUPRC | Precision | Recall | F1 Score |
|---|---|---:|---:|---:|---:|---:|
| **$K=2$ frames** | $\approx 66\text{ ms}$ | **0.8174** | 0.6370 | 0.3407 | **0.8057** | 0.4789 |
| **$K=3$ frames** | $\approx 99\text{ ms}$ | **0.8052** | 0.6560 | 0.4284 | **0.7729** | 0.5512 |
| **$K=5$ frames (Primary)** | $\approx 165\text{ ms}$ | **0.8046** | **0.7260** | **0.5808** | 0.7336 | **0.6484** |

### 3. Causal Streaming Replay Evaluation (10,481 Frames)
Evaluating the frozen predictor in causal sequential replay using a strict rolling FIFO buffer ($N=6$):

| Evaluation Dimension | Measurement | Verification & Detail |
|---|---|---|
| **Classification Integrity** | AUROC = **0.8057**, F1 = **0.6483** | Replay matches offline test evaluation |
| **Feature Construction Latency** | Mean: **$3.8\,\mu\text{s}$** (P99: $8.1\,\mu\text{s}$) | Rolling circular buffer arithmetic |
| **StandardScaler Latency** | Mean: **$80.1\,\mu\text{s}$** (P99: $131.0\,\mu\text{s}$) | In-place NumPy affine scaling |
| **MLP Forward Pass Latency** | Mean: **$43.3\,\mu\text{s}$** (P99: $74.1\,\mu\text{s}$) | PyTorch CPU forward inference |
| **Total End-to-End Latency** | Mean: **0.1282 ms** (Median: 0.1213 ms, P99: 0.2139 ms) | **99.36% of 33.3 ms budget remains free** |
| **Total Failure Episodes** | 658 contiguous episodes | Grouped consecutive ground-truth failure sequences |
| **Advance Warning Rate** | **522 episodes (79.33%)** | Emitted positive alarm *prior to* failure onset |
| **Advance Lead Time** | **Median: 0.165 s** (Mean: 0.153 s) | Full 5-frame forward window anticipation |

### 4. Real-Time ROS 2 Middleware Validation
The predictor was deployed as an active ROS 2 node (`LiveFailurePredictorNode`) communicating via standard ROS 2 message types (`geometry_msgs/msg/Vector3Stamped` for telemetry ingestion on `/telemetry/motion` and `geometry_msgs/msg/PointStamped` for predictions on `/vo/failure_prediction`) on the flight condition `sweep_A_C_R1` ($45.0^\circ/\text{s}$ commanded yaw rate):
- **Streaming Throughput**: Sustained inference up to **192 Hz** (far exceeding the 30 Hz sensor rate).
- **Middleware Execution Latency**: Mean **0.6335 ms** (P99 **3.7128 ms**), preserving >88% timing headroom.
- **Live Advance Warning Rate**: **71 / 76 failure episodes (93.42%)** received advance warnings prior to onset, with a median lead time of **0.165 s**.

---

## Reproducibility & Audit Verification

Exhaustive replication instructions and artifact hashes are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

```bash
# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run complete 7-stage forensic audit
python3 scripts/audit_final/01_dataset_and_failure_forensics.py
python3 scripts/audit_final/02_label_and_leakage_forensics.py
python3 scripts/audit_final/03_models_and_baselines_forensics.py
python3 scripts/audit_final/04_early_warning_and_alarms.py
python3 scripts/audit_final/05_uncertainty_and_generalization.py
python3 scripts/audit_final/06_causal_replay_and_ros2_forensics.py
python3 scripts/audit_final/07_synthesis_and_claim_matrix.py

# Independent verification script
python3 scripts/audit_final/verify_independent.py

# Run unit tests
pytest tests/test_causal_leakage.py
```

---

## Citation

If you use this work, codebase, or dataset in your research, please cite:

```bibtex
@article{sen2026aegis,
  author    = {Sen, Purab},
  title     = {AEGIS: Telemetry-Based Early Warning of Monocular Visual Odometry Tracking Dropouts During Aggressive UAV Flight},
  year      = {2026},
  doi       = {10.5281/zenodo.23267668},
  url       = {https://doi.org/10.5281/zenodo.23267668},
  publisher = {Zenodo}
}
```

- **Author**: Purab Sen ([ORCID: 0009-0002-8385-1435](https://orcid.org/0009-0002-8385-1435)), Independent Researcher
- **Concept DOI**: [10.5281/zenodo.23267668](https://doi.org/10.5281/zenodo.23267668)

---

## License

- **Code License**: [Apache License 2.0](LICENSE) (see `LICENSE`)
- **Data, Figures, and Paper Text**: [Creative Commons Attribution 4.0 International](LICENSE-DATA.md) (CC BY 4.0, see `LICENSE-DATA.md`)
