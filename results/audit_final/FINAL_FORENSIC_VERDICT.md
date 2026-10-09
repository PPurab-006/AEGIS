# AEGIS Final Forensic Verdict

**Standard**: TRUTH > CONSISTENCY > COMPLETENESS > PRESENTATION  
**Date of Audit**: October 2026  
**Repository State**: Git commit `08ec3ccef2d95edf8072e64998be4f1447809d49`  

---

### A. What was wrong with the original AEGIS result?
Four primary flaws undermined the original claims:
1. **Contemporaneous Feature Leakage**: The legacy target $[t, t+K]$ included the current frame $t$. Because optical flow drops to near zero (< 0.5 px) during tracking failure with 97.3% probability, the model was essentially acting as a co-occurring failure detector on already-failed frames, artificially inflating test AUROC from 0.7665 to 0.8046 (+0.0381, $p < 0.001$) and AUPRC from 0.5977 to 0.7260 (+0.1283, $p < 0.001$).
2. **Failure Mechanism Misrepresentation**: The paper described events as "catastrophic VO collapse", whereas 99.59% of events are isolated single-frame (33.3 ms) tracking skips caused by feature re-detection in `minimal_vo.py`.
3. **Unsupportable Machine Learning Claims**: The paper claimed MLP superiority, but standard gradient-boosted decision trees (HistGradientBoosting) outperform the MLP (AUROC 0.7733 vs 0.7665; AUPRC 0.6101 vs 0.5982).
4. **Alarm Fatigue Obfuscation**: The paper reported high detection rates without emphasizing that at native operating thresholds, the model triggers ~26 false alarms per minute and suffers an alarm duty cycle of ~41%.

---

### B. What remains valid?
1. **Statistical Anticipation Exists**: Telemetry contains genuine physical signals (elevated yaw rates and optical flow history) that precede tracking dropouts. Strict future-only prediction achieves **0.7665 AUROC** and **0.5982 AUPRC** (vs 0.3024 baseline), significantly above chance ($p < 0.001$).
2. **Event-Level Early Warning**: Over 83% of failure episodes are preceded by an alarm within 0.50 s, with median lead times of ~0.37 s.
3. **Computational Feasibility**: The streaming inference pipeline executes in < 1 ms in a multi-process ROS 2 Humble environment with zero message drops across thousands of frames.
4. **Geometry Generalization**: The predictor generalizes well across unseen trajectory shapes (LOGO mean AUROC 0.7688).

---

### C. What is the strongest defensible scientific result?
Kinematic and optical flow telemetry passively collected from a drone platform contains predictive information capable of forecasting monocular visual-odometry tracking dropouts up to 0.50 s in advance under strict future-only evaluation (AUROC 0.7665, AUPRC 0.5982).

---

### D. What is the strongest defensible systems result?
The streaming failure predictor operates with deterministic low latency (< 1.0 ms callback compute, < 2.0 ms total round-trip P99 latency) inside a multi-process ROS 2 architecture, with zero message loss observed through 500 Hz on the evaluation host.

---

### E. What claims must be removed?
1. **REMOVE**: Claims that AEGIS predicts "prolonged VO estimator collapse / drift".
2. **REMOVE**: Claims that SmallMLP is architecturally superior to classical estimators.
3. **REMOVE**: Claims that AEGIS generalizes across operational rotational velocities (LOYO AUROC collapses to 0.6466).
4. **REMOVE**: Claims of legacy AUROC 0.8046.
5. **REMOVE**: Claims that 0.165 s is the physical upper bound of anticipation.
6. **REMOVE**: Claims that AEGIS is ready for unaugmented closed-loop autonomous flight deployment.

---

### F. What claims can remain?
1. Passive telemetry contains statistical anticipation of impending visual-odometry dropouts.
2. Strict future-only prediction functions with AUROC 0.7665 and AUPRC 0.5982.
3. Optical flow history is an essential predictive feature (removing flow features drops AUROC to 0.7199).
4. Streaming causal replay matches batch PyTorch inference to within $10^{-6}$ precision.
5. The inference pipeline is computationally practical in ROS 2.

---

### G. What claims need qualification?
1. **Real-time capacity**: State that zero drops were observed up to 500 Hz on the desktop host, but qualify that delivery burst rates were bounded to ~206 Hz by OS timer resolution and embedded hardware remains unbenchmarked.
2. **Generalization**: State that the model generalizes across trajectory geometries (0.7688 AUROC) and repeated flight cells, but explicitly acknowledge that performance degrades under held-out yaw rates (0.6466 AUROC).
3. **Operational Utility**: State that early warning occurs for > 80% of events, but qualify that native false alarm rates (~26/min) necessitate temporal filtering before closed-loop control integration.

---

### H. What are the three biggest scientific limitations?
1. **Benign Single-Frame Failure Mode**: The benchmark evaluates single-frame re-detection dropouts rather than divergent visual-inertial state divergence.
2. **Alarm Burden & False Positive Rate**: An alarm duty cycle of ~40% and false alarm rate of ~26 edges/min imposes prohibitive alarm fatigue in continuous operation.
3. **Rotational Domain Sensitivity**: The predictor relies heavily on yaw-rate thresholds, degrading substantially when tested on unseen rotational velocities.

---

### I. What is the single best one-sentence conclusion for AEGIS?
"While passive kinematic and visual telemetry provides statistically genuine early warning of monocular tracking dropouts up to 0.5 seconds in advance (strict AUROC 0.7665), practical deployment is constrained by high alarm burden and sensitivity to unseen rotational velocities, demonstrating the viability of telemetry-based anticipation while establishing that robust autonomy requires operational false-alarm mitigation."
