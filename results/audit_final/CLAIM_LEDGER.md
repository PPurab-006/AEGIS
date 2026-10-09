# AEGIS Scientific Claim Ledger (Final Post-Audit Ledger)

Standard: **TRUTH > CONSISTENCY > COMPLETENESS > PRESENTATION**

This document establishes the definitive classification of every major scientific and systems claim that could appear in the revised AEGIS preprint.

---

### Claim 1: Telemetry contains anticipatory information.
- **Classification**: **SUPPORTED**
- **Evidence**: On strictly held-out non-failed frames predicting strictly future failures in $[t+1, t+5]$, models using past kinematic and visual telemetry achieve AUROC of **0.7665** (AUPRC **0.5982** vs base rate **0.3024**). At $W=15$ (0.50 s pre-onset), 83.15% of failure episodes are preceded by an active alarm, with a median lead time of **0.367 s**.
- **Qualification / Safe Wording**: Telemetry contains genuine statistical anticipation of impending estimator tracking failure up to 0.50 s in advance, significantly exceeding random chance.

---

### Claim 2: Strict future-only prediction works above chance.
- **Classification**: **SUPPORTED**
- **Evidence**: With contemporaneous failure frames excluded and target restricted strictly to $[t+1, t+5]$, both frozen MLP (AUROC **0.7665**, 95% CI [0.7140, 0.8053]) and retrained strict V0 (AUROC **0.7665**, 95% CI [0.7150, 0.8039]) substantially outperform chance (AUROC 0.50) and matched-duty random baselines (AUPRC 0.3024).
- **Qualification / Safe Wording**: Predictive performance remains above chance under strict future-only evaluation, although AUROC is lower than the originally published 0.8046 legacy metric.

---

### Claim 3: Lag-0 leakage materially affects the legacy evaluation.
- **Classification**: **SUPPORTED**
- **Evidence**: In legacy evaluation, target $[t, t+K]$ included the current frame $t$. On failed frames, `feature_vel_mean_lag0` drops below 0.5 with probability **97.32%**, and $P(\text{failure} \mid \text{flow} < 0.5) = 97.08\%$. The frozen model scored positive on **98.32% (642/653)** of currently failed test frames, mechanically inflating legacy True Positives from 2,002 to 2,644. In paired flight bootstrap, legacy AUROC exceeds strict AUROC by **+0.0387** ($p < 0.001$) and AUPRC by **+0.1289** ($p < 0.001$).
- **Qualification / Safe Wording**: The legacy evaluation protocol suffered from contemporaneous feature leakage that materially inflated discrimination metrics by counting co-occurring detection of existing failures as early warnings.

---

### Claim 4: Removing lag-0 flow destroys predictive performance.
- **Classification**: **NOT SUPPORTED**
- **Evidence**: In strict training where $t$ is non-failed and target is $[t+1, t+5]$, removing `feature_vel_mean_lag0` (Variant V1) yields AUROC **0.7669** and AUPRC **0.6002**, which is virtually identical to Variant V0 (AUROC **0.7665**, AUPRC **0.5982**). However, removing *all* flow features (Variant V2) causes AUROC to drop sharply to **0.7199** and AUPRC to **0.4911**.
- **Qualification / Safe Wording**: Removing contemporaneous flow (lag-0) does not degrade strict predictive performance because lagged flow history (lags 1-5) and yaw dynamics provide redundant predictive information. However, removing optical flow features entirely causes a substantial ~0.047 AUROC drop.

---

### Claim 5: MLP is superior to classical models.
- **Classification**: **NOT SUPPORTED**
- **Evidence**: HistGradientBoosting achieves AUROC **0.7733** and AUPRC **0.6101** on the strict test set, compared to SmallMLP's AUROC **0.7665** and AUPRC **0.5982**. In matched 30% duty cycle evaluation ($W=15$), HGB achieves **79.13%** rising-edge early warnings vs V0's **62.91%**, with fewer spurious alarms (11.3 vs 7.7 rising edges/min).
- **Qualification / Safe Wording**: The claim of neural network (MLP) superiority is unsupported. A standard decision-tree ensemble (HistGradientBoosting) matches or slightly exceeds MLP discrimination on the tabular telemetry feature set.

---

### Claim 6: HGB is superior to MLP.
- **Classification**: **SUPPORTED WITH QUALIFICATION**
- **Evidence**: HGB obtains higher AUROC (0.7733 vs 0.7665) and higher AUPRC (0.6101 vs 0.5982) than MLP V0. At fixed threshold 0.50, HGB is more conservative (21.2% duty cycle) while retaining strong precision (61.4%).
- **Qualification / Safe Wording**: HistGradientBoosting exhibits slightly superior discriminative performance over SmallMLP on tabular offline evaluation (+0.0068 AUROC, +0.0119 AUPRC), although both share similar operational false-alarm constraints and computational feasibility.

---

### Claim 7: Model generalizes across repeated flight cells.
- **Classification**: **SUPPORTED**
- **Evidence**: On the standard repeat-held-out test split (14 flights across identical trajectory/yaw cells), the model achieves AUROC **0.7665** (seed 42) and **0.7681 ± 0.0024** across 5 random seeds.
- **Qualification / Safe Wording**: The model reliably generalizes to unseen physical flight executions conducted within previously seen trajectory geometry and rotational rate operating regimes.

---

### Claim 8: Model generalizes across yaw rates.
- **Classification**: **NOT SUPPORTED**
- **Evidence**: Leave-One-Yaw-Rate-Out (LOYO) cross-validation exhibits severe degradation: mean AUROC drops to **0.6466 ± 0.0423**, falling as low as **0.5949** when moderate/aggressive yaw rates are held out.
- **Qualification / Safe Wording**: The model does NOT generalize across unseen yaw-rate regimes. Performance degrades substantially under held-out rotational velocities.

---

### Claim 9: Model generalizes across trajectory geometries.
- **Classification**: **SUPPORTED WITH QUALIFICATION**
- **Evidence**: Leave-One-Geometry-Out (LOGO) cross-validation maintains a mean AUROC of **0.7688 ± 0.0319** (range 0.7342 to 0.8109), comparable to the standard test split.
- **Qualification / Safe Wording**: The predictor generalizes effectively across trajectory shapes (circles, figure-8s, stars, handheld paths) provided the training distribution encompasses the operational rotational velocities.

---

### Claim 10: Failure events represent prolonged VO collapse.
- **Classification**: **NOT SUPPORTED**
- **Evidence**: Trace analysis of `minimal_vo.py` and dataset ground truth across 31,615 active frames reveals that **99.59% (1,961 of 1,969)** of failure episodes are exactly **1 frame long (33.3 ms)**. Maximum failure duration in the entire dataset is 3 frames (100 ms).
- **Qualification / Safe Wording**: The failures in this benchmark are not prolonged estimator divergent crashes; they are transient single-frame tracking dropouts during feature re-detection where pose estimation is skipped for a single frame.

---

### Claim 11: Failure events are predominantly single-frame dropouts.
- **Classification**: **SUPPORTED**
- **Evidence**: Empirical failure episode length distribution across 42 flights: length 1 = 1,961 episodes (99.5937%), length 2 = 7 episodes (0.3555%), length 3 = 1 episode (0.0508%).
- **Qualification / Safe Wording**: Verified from repository source code and raw data: over 99.5% of visual odometry tracking dropouts are single-frame events caused by feature re-detection resets.

---

### Claim 12: Early warning can precede failure.
- **Classification**: **SUPPORTED**
- **Evidence**: With pre-onset anticipation window $W=15$ frames (0.50 s), SmallMLP V0 generates active alarms prior to failure onset for **83.15%** of episodes, with a median lead time of **0.367 s** and mean lead time of **0.337 s**. At $W=30$ (1.00 s), warning occurs for **85.94%** of episodes with median lead time **0.767 s**.
- **Qualification / Safe Wording**: Telemetry-based warning signals reliably precede failure onsets by 0.3 to 0.8 seconds.

---

### Claim 13: Early warning is operationally useful.
- **Classification**: **NOT SUPPORTED (AS CURRENTLY CONFIGURED)**
- **Evidence**: At native operating thresholds ($	heta^* = 0.54$), Retrained V0 incurs an alarm duty cycle of **41.16%** and triggers **107.17 alarm rising edges per minute**, of which **25.93 per minute are false alarms**. This corresponds to a spurious alarm every 2.3 seconds of flight time.
- **Qualification / Safe Wording**: While temporal anticipation is statistically present, the native alarm burden (~41% duty cycle, ~26 false alarm bursts/min) is operationally prohibitive for unaugmented flight autonomy without downstream temporal filtering, hysteresis, or multi-frame consensus.

---

### Claim 14: System is computationally feasible in ROS2.
- **Classification**: **SUPPORTED**
- **Evidence**: In a multi-process ROS 2 Humble deployment across 4 real flights (2,991 frames at 30 Hz), zero message drops were observed (0.00% drop rate). Predictor callback compute time had a median of **0.56 - 0.76 ms** (P99 **0.79 - 1.01 ms**), and end-to-end round-trip latency was median **1.20 - 1.51 ms** (P99 **1.62 - 2.00 ms**), consuming < 6% of the 33.3 ms frame period.
- **Qualification / Safe Wording**: The inference pipeline is computationally lightweight and operates comfortably within a 30 Hz ROS 2 execution budget on desktop hardware.

---

### Claim 15: 500 Hz real-time capacity is established.
- **Classification**: **SUPPORTED WITH QUALIFICATION**
- **Evidence**: Stress testing through 500 Hz nominal publication rates yielded 400/400 received messages without message loss. However, non-real-time Linux sleep resolution bounded actual delivered burst rates to ~206 Hz.
- **Qualification / Safe Wording**: No message drops were observed through the highest tested rate of 500 Hz on the evaluation host. Deployment on resource-constrained embedded UAV hardware remains unverified.

---

### Claim 16: AEGIS is suitable for autonomous deployment.
- **Classification**: **NOT SUPPORTED (WITHOUT REFINEMENT)**
- **Evidence**: High false-alarm rates (25.9/min), single-frame failure mechanism mismatch, and poor out-of-distribution yaw generalization preclude direct, unaugmented closed-loop flight intervention.
- **Qualification / Safe Wording**: AEGIS demonstrates the scientific feasibility of telemetry-driven early warning, but operational autonomous deployment requires mitigating alarm fatigue, filtering single-frame dropouts, and broadening yaw-rate training distributions.
