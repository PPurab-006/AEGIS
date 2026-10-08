# Research 2 Publication Figures — Manuscript Caption Drafts

This document contains publication-ready caption drafts for the 5 target figures created for the Research 2 preprint (*Learned Telemetry-Based Prediction of Monocular Visual Odometry Failure*).

---

## Figure 1: Research Pipeline & Experimental Design

**File paths**:
- Vector PDF: [`results/figures/publication/fig1_pipeline_schematic.pdf`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig1_pipeline_schematic.pdf)
- Vector SVG: [`results/figures/publication/fig1_pipeline_schematic.svg`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig1_pipeline_schematic.svg)
- Raster 300 DPI PNG: [`results/figures/publication/fig1_pipeline_schematic.png`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig1_pipeline_schematic.png)

### Draft Caption:
> **Figure 1. End-to-end research methodology, data partitioning, and multi-tier verification pipeline.** **(a)** Data generation and leak-free quarantine: A parametric $4 \times 4$ flight sweep combining four commanded body yaw rates (Gentle $7.5^\circ/\text{s}$, Moderate $22.5^\circ/\text{s}$, Aggressive $45.0^\circ/\text{s}$, Extreme $90.0^\circ/\text{s}$) across four trajectory geometries (Circle, Box, Straight line, Helix) was executed in PX4 Software-In-The-Loop simulation with Gazebo Sim. After strict physical flight envelope filtering (excluding 6 unviable trials), 42 eligible flights totaling 31,195 frames (34.6% failure prevalence) were partitioned strictly at the flight level (Seed 42) into disjoint training (15 flights, 11,178 frames), validation (13 flights, 9,606 frames), and held-out test (14 flights, 10,411 frames) sets ($\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). **(b)** Causal inference pipeline: Vehicle motion telemetry streamed at 30 Hz (body yaw rate $\omega_z$, optical flow mean displacement $\bar{v}_{\text{flow}}$, and keyframe flag $r$) feeds a rolling FIFO buffer ($N=6$, 165 ms history) to construct a 15-dimensional lagged feature vector $x_t$ over backward lags $\{0, 1, 2, 3, 5\}$. A frozen StandardScaler (fit exclusively on the training split) normalizes features for a compact Multi-Layer Perceptron (`SmallMLP`: $15 \to 32 \to 16 \to 1$, 1,057 parameters, $0.128\text{ ms}$ CPU latency) emitting failure probability $P(\text{fail}_{t+5} \mid x_t)$ for forward horizon $K=5$ frames ($\approx 165\text{ ms}$). An operational decision threshold ($\theta = 0.50$) triggers early warning alerts. **(c)** Three-tier empirical verification framework: Tier 1 evaluates offline generalization across the 14 held-out test flights (AUROC 0.8046, AUPRC 0.7260); Tier 2 verifies causal sequential replay across 10,481 frames in timestamp order with zero future access (79.3% true advance warnings, median lead time 0.165 s); Tier 3 validates real-time ROS 2 middleware execution (`LiveFailurePredictorNode`, 192 Hz sustained rate, $0.634\text{ ms}$ mean latency, 93.4% live advance warning rate).

---

## Figure 2: Main Held-Out Predictive Performance

**File paths**:
- Vector PDF: [`results/figures/publication/fig2_held_out_performance.pdf`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig2_held_out_performance.pdf)
- Vector SVG: [`results/figures/publication/fig2_held_out_performance.svg`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig2_held_out_performance.svg)
- Raster 300 DPI PNG: [`results/figures/publication/fig2_held_out_performance.png`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig2_held_out_performance.png)

### Draft Caption:
> **Figure 2. Held-out predictive performance on the 14-flight test partition ($N = 10,411$ frames).** **(a)** Receiver Operating Characteristic (ROC) curve: The frozen `SmallMLP` achieves an AUROC of 0.8046 on unseen flight trajectories, substantially outperforming chance (0.5000, dotted line). The red marker indicates the locked operational operating point ($\theta = 0.50$), yielding a True Positive Rate (Recall) of 0.7336 and False Positive Rate of 0.2803. **(b)** Precision-Recall (PR) trajectory: The model achieves an Area Under the Precision-Recall Curve (AUPRC) of 0.7260, more than doubling the test set positive failure prevalence baseline (0.3462, dotted line). At $\theta = 0.50$, the classifier operates at Precision 0.5808, Recall 0.7336, and F1 score 0.6484. **(c)** Operational threshold sensitivity trade-off: Precision, Recall, and F1 score evaluated across candidate decision thresholds $\theta \in [0.10, 0.90]$. The locked threshold ($\theta = 0.50$, vertical dotted red line) aligns closely with the empirical F1 optimum without requiring post-hoc flight-specific tuning.

---

## Figure 3: Temporal Early-Warning Behavior

**File paths**:
- Vector PDF: [`results/figures/publication/fig3_temporal_early_warning.pdf`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig3_temporal_early_warning.pdf)
- Vector SVG: [`results/figures/publication/fig3_temporal_early_warning.svg`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig3_temporal_early_warning.svg)
- Raster 300 DPI PNG: [`results/figures/publication/fig3_temporal_early_warning.png`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig3_temporal_early_warning.png)

### Draft Caption:
> **Figure 3. Temporal early-warning dynamics during causal streaming execution.** Representative 0.60-second flight window ($t = 28.45\text{ s}$ to $29.05\text{ s}$) from held-out test flight `sweep_A_C_R1` (commanded body yaw rate $45^\circ/\text{s}$), demonstrating true advance warning prior to tracking collapse. **(a)** Onboard motion telemetry streams: Body yaw rate $\omega_z$ (solid navy line with circle markers) and mean optical flow velocity $\bar{v}_{\text{flow}}$ (dash-dotted teal line with triangle markers) sampled at 30 Hz. **(b)** Predicted failure probability: The model probability $P(\text{fail}_{t+5} \mid x_t)$ (purple line with square markers) rises steadily and crosses the locked threshold $\theta = 0.50$ at $t = 28.744\text{ s}$ (warning onset, orange vertical dashed guide), initiating an active warning interval (amber shaded region) that persists across 5 consecutive frames. **(c)** Visual Odometry solver tracking state: Monocular VO inlier count $N_{\text{inliers}}$ (charcoal line with diamond markers) remains healthy ($N = 84\text{--}144$, well above the failure threshold $N_{\text{fail}} = 8$, red dashed line) throughout the entire warning window. At $t = 28.909\text{ s}$, epipolar geometry degenerates and inlier count collapses to 0 ($< 8$, red shaded region), confirming failure onset. The warning preceded solver collapse by exactly $\Delta t = 0.165\text{ s}$ (5 frames @ 30 Hz), matching the median advance lead time across all 522 early warning episodes in the 14-flight test evaluation. Tracking subsequently recovers at $t = 28.942\text{ s}$ ($N = 1,235$), returning probability below threshold ($P = 0.385$).

---

## Figure 4: Cross-Flight Generalization

**File paths**:
- Vector PDF: [`results/figures/publication/fig4_cross_flight_generalization.pdf`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig4_cross_flight_generalization.pdf)
- Vector SVG: [`results/figures/publication/fig4_cross_flight_generalization.svg`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig4_cross_flight_generalization.svg)
- Raster 300 DPI PNG: [`results/figures/publication/fig4_cross_flight_generalization.png`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig4_cross_flight_generalization.png)

### Draft Caption:
> **Figure 4. Cross-flight generalization across all 14 disjoint held-out test flights ($K=5$, $\theta = 0.50$).** **(a)** Individual flight AUROC distribution: Point-stem plot of AUROC scores ranked from lowest to highest, color-coded by commanded yaw rate dynamics: Gentle (green, $7.5^\circ/\text{s}$), Moderate (blue, $22.5^\circ/\text{s}$), Aggressive (orange, $45.0^\circ/\text{s}$), and Extreme (red, $90.0^\circ/\text{s}$). Every single held-out flight achieves performance substantially above the 0.5000 chance baseline (solid gray line), spanning from 0.6690 (`sweep_M_S_R2`) to 0.8393 (`sweep_A_H_R2`). The pooled test AUROC (0.8046, dashed red line) and unweighted mean flight AUROC (0.7606, dotted orange line) demonstrate robust generalization across varied trajectory geometries (Circle, Box, Helix, Straight). **(b)** Per-flight Precision, Recall, and F1 trade-off profiles: Horizontal grouped bars showing Precision (blue), F1 score (green), and Recall (orange) for each test flight alongside pooled test benchmarks (F1 0.6484, Precision 0.5808, Recall 0.7336). On gentle flights with lower failure prevalence (11–14%), precision reaches up to 0.7091, whereas on aggressive and extreme flights with severe tracking degradation (up to 69.5% failure prevalence), recall reaches up to 0.9519, confirming consistent discriminative capacity across dynamic regimes.

---

## Figure 5: Real-Time Feasibility & ROS 2 Middleware Validation

**File paths**:
- Vector PDF: [`results/figures/publication/fig5_realtime_feasibility.pdf`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig5_realtime_feasibility.pdf)
- Vector SVG: [`results/figures/publication/fig5_realtime_feasibility.svg`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig5_realtime_feasibility.svg)
- Raster 300 DPI PNG: [`results/figures/publication/fig5_realtime_feasibility.png`](file:///home/purab/Purab/Projects/Research2/results/figures/publication/fig5_realtime_feasibility.png)

### Draft Caption:
> **Figure 5. Real-time computational feasibility and live ROS 2 middleware latency validation.** **(a)** Algorithmic stage latency breakdown: Profiling across 10,481 streaming inference cycles in sequential causal replay. Mean latency (light blue) and 99th percentile latency (dark navy) are shown for feature FIFO buffering (mean $3.8\,\mu\text{s}$, P99 $8.1\,\mu\text{s}$), StandardScaler normalization (mean $80.1\,\mu\text{s}$, P99 $131.0\,\mu\text{s}$), and PyTorch SmallMLP forward pass (mean $43.3\,\mu\text{s}$, P99 $74.1\,\mu\text{s}$). Total algorithmic end-to-end latency averages $128.2\,\mu\text{s}$ ($0.128\text{ ms}$, bold blue annotation; P99 $213.9\,\mu\text{s}$ / $0.214\text{ ms}$, bold navy annotation), consuming less than 1% of the 33.3 ms sample period. **(b)** Live ROS 2 middleware execution latency distribution: Empirical probability density of end-to-end callback execution latency across 754 live prediction messages processed by `LiveFailurePredictorNode` subscribing to `/telemetry/motion` and publishing to `/vo/failure_prediction`. The distribution exhibits a median of $0.457\text{ ms}$ (solid green line), mean of $0.634\text{ ms}$ (dashed blue line), 95th percentile of $1.008\text{ ms}$, and 99th percentile of $3.713\text{ ms}$ (dotted red line), preserving $>88.8\%$ headroom relative to the $33.33\text{ ms}$ sensor period. **(c)** Operational throughput and timing margin: Comparison of required sensor stream ingestion rate ($30.0\text{ Hz}$, navy bar) against sustained processing capacity ($191.97\text{ Hz}$, teal bar), demonstrating a $6.4\times$ computational throughput margin that confirms the predictor is computationally negligible and viable for real-time onboard deployment.
