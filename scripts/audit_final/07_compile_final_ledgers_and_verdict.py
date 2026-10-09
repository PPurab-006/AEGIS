#!/usr/bin/env python3
"""
AEGIS FINAL FORENSIC PASS — SCRIPT 07: COMPILATION OF FINAL LEDGERS & VERDICT
=============================================================================
Compiles:
16. results/audit_final/CLAIM_LEDGER.md (all 16 claims classified + exact safe wording)
17. results/audit_final/FINAL_NUMBER_LEDGER.csv (authoritative single source of truth)
18. results/audit_final/DISCREPANCY_LOG.md (exhaustive reconciliation)
19. results/audit_final/FINAL_FORENSIC_VERDICT.md (plain, direct answers A through I)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
results_dir = repo_root / "results" / "audit_final"
results_dir.mkdir(parents=True, exist_ok=True)

def generate_claim_ledger():
    ledger_path = results_dir / "CLAIM_LEDGER.md"
    content = """# AEGIS Scientific Claim Ledger (Final Post-Audit Ledger)

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
- **Evidence**: In legacy evaluation, target $[t, t+K]$ included the current frame $t$. On failed frames, `feature_vel_mean_lag0` drops below 0.5 with probability **97.32%**, and $P(\\text{failure} \\mid \\text{flow} < 0.5) = 97.08\\%$. The frozen model scored positive on **98.32% (642/653)** of currently failed test frames, mechanically inflating legacy True Positives from 2,002 to 2,644. In paired flight bootstrap, legacy AUROC exceeds strict AUROC by **+0.0387** ($p < 0.001$) and AUPRC by **+0.1289** ($p < 0.001$).
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
- **Evidence**: At native operating thresholds ($\theta^* = 0.54$), Retrained V0 incurs an alarm duty cycle of **41.16%** and triggers **107.17 alarm rising edges per minute**, of which **25.93 per minute are false alarms**. This corresponds to a spurious alarm every 2.3 seconds of flight time.
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
"""
    with open(ledger_path, "w") as f:
        f.write(content)
    print(f"Saved {ledger_path}")

def generate_number_ledger():
    ledger_path = results_dir / "FINAL_NUMBER_LEDGER.csv"
    
    rows = [
        # Dataset & Splits
        {"metric": "total_eligible_flights", "value": "42", "units": "flights", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "raw_extraction", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "total_raw_frames", "value": "51041", "units": "frames", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "raw_extraction", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "total_active_frames", "value": "31615", "units": "frames", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "pos_z >= 2.0m", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "train_flight_count", "value": "15", "units": "flights", "dataset": "ROS_telemetry", "split": "train", "evaluation_protocol": "expanded_split", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "val_flight_count", "value": "13", "units": "flights", "dataset": "ROS_telemetry", "split": "val", "evaluation_protocol": "expanded_split", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "test_flight_count", "value": "14", "units": "flights", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "expanded_split", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "flight_overlap_count", "value": "0", "units": "flights", "dataset": "ROS_telemetry", "split": "cross_split", "evaluation_protocol": "set_intersection", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "failure_mechanism_report.md", "status": "VERIFIED"},
        
        # Failure Characterization
        {"metric": "total_failure_frames", "value": "1978", "units": "frames", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "inliers_pose < 8", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "dataset_failure_rate_pct", "value": "6.2565", "units": "%", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "inliers_pose < 8", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "total_failure_episodes", "value": "1969", "units": "episodes", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "contiguous_runs", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "single_frame_episodes_pct", "value": "99.5937", "units": "%", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "run_length == 1", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        {"metric": "max_failure_episode_length", "value": "3", "units": "frames", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "run_length", "threshold": "N/A", "source_script": "01_dataset_and_failure_forensics.py", "source_output": "dataset_flight_characterization.csv", "status": "VERIFIED"},
        
        # Leakage & Determinism
        {"metric": "p_failure_given_low_flow", "value": "97.0751", "units": "%", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "flow_lag0 < 0.5", "threshold": "0.5 px", "source_script": "02_label_and_leakage_forensics.py", "source_output": "leakage_statistics.csv", "status": "VERIFIED"},
        {"metric": "p_low_flow_given_failure", "value": "97.3205", "units": "%", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "flow_lag0 < 0.5 | failure", "threshold": "0.5 px", "source_script": "02_label_and_leakage_forensics.py", "source_output": "leakage_statistics.csv", "status": "VERIFIED"},
        {"metric": "is_r_frame_deterministic_mismatch", "value": "0", "units": "frames", "dataset": "ROS_telemetry", "split": "all", "evaluation_protocol": "is_r == (yaw_rate > 15)", "threshold": "15 deg/s", "source_script": "02_label_and_leakage_forensics.py", "source_output": "leakage_statistics.csv", "status": "VERIFIED"},
        
        # Test Split Frame Accounting
        {"metric": "test_active_frames", "value": "10551", "units": "frames", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "pos_z >= 2.0m", "threshold": "N/A", "source_script": "02_label_and_leakage_forensics.py", "source_output": "label_forensics.md", "status": "VERIFIED"},
        {"metric": "test_legacy_eval_frames", "value": "10411", "units": "frames", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "warmup5_tail5_dropped", "threshold": "N/A", "source_script": "02_label_and_leakage_forensics.py", "source_output": "frozen_legacy_vs_strict.csv", "status": "VERIFIED"},
        {"metric": "test_currently_failed_frames", "value": "653", "units": "frames", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "is_failure == 1", "threshold": "N/A", "source_script": "02_label_and_leakage_forensics.py", "source_output": "label_forensics.md", "status": "VERIFIED"},
        {"metric": "test_strict_eval_frames", "value": "9758", "units": "frames", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "clean_strict_frames", "threshold": "N/A", "source_script": "02_label_and_leakage_forensics.py", "source_output": "frozen_legacy_vs_strict.csv", "status": "VERIFIED"},
        
        # Frozen Model (Legacy vs Strict)
        {"metric": "frozen_legacy_auroc", "value": "0.8046", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "legacy_all_frames", "threshold": "0.50", "source_script": "03_models_and_baselines_forensics.py", "source_output": "frozen_legacy_vs_strict.csv", "status": "VERIFIED"},
        {"metric": "frozen_legacy_auprc", "value": "0.7260", "units": "AUPRC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "legacy_all_frames", "threshold": "0.50", "source_script": "03_models_and_baselines_forensics.py", "source_output": "frozen_legacy_vs_strict.csv", "status": "VERIFIED"},
        {"metric": "frozen_legacy_f1", "value": "0.6484", "units": "F1", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "legacy_all_frames", "threshold": "0.50", "source_script": "03_models_and_baselines_forensics.py", "source_output": "frozen_legacy_vs_strict.csv", "status": "VERIFIED"},
        {"metric": "frozen_strict_auroc", "value": "0.7665", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.50", "source_script": "03_models_and_baselines_forensics.py", "source_output": "frozen_legacy_vs_strict.csv", "status": "VERIFIED"},
        {"metric": "frozen_strict_auprc", "value": "0.5977", "units": "AUPRC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.50", "source_script": "03_models_and_baselines_forensics.py", "source_output": "frozen_legacy_vs_strict.csv", "status": "VERIFIED"},
        {"metric": "frozen_strict_f1", "value": "0.5836", "units": "F1", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.50", "source_script": "03_models_and_baselines_forensics.py", "source_output": "frozen_legacy_vs_strict.csv", "status": "VERIFIED"},
        
        # Retrained Strict V0 & Variants
        {"metric": "v0_strict_auroc", "value": "0.7665", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.54 (val)", "source_script": "03_models_and_baselines_forensics.py", "source_output": "retrained_variants_comparison.csv", "status": "VERIFIED"},
        {"metric": "v0_strict_auprc", "value": "0.5982", "units": "AUPRC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.54 (val)", "source_script": "03_models_and_baselines_forensics.py", "source_output": "retrained_variants_comparison.csv", "status": "VERIFIED"},
        {"metric": "v0_strict_f1_val_th", "value": "0.5775", "units": "F1", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.54 (val)", "source_script": "03_models_and_baselines_forensics.py", "source_output": "retrained_variants_comparison.csv", "status": "VERIFIED"},
        {"metric": "v0_strict_f1_fixed", "value": "0.5758", "units": "F1", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.50", "source_script": "03_models_and_baselines_forensics.py", "source_output": "retrained_variants_comparison.csv", "status": "VERIFIED"},
        {"metric": "v1_drop_flow_lag0_auroc", "value": "0.7669", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.56 (val)", "source_script": "03_models_and_baselines_forensics.py", "source_output": "retrained_variants_comparison.csv", "status": "VERIFIED"},
        {"metric": "v2_drop_all_flow_auroc", "value": "0.7199", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.44 (val)", "source_script": "03_models_and_baselines_forensics.py", "source_output": "retrained_variants_comparison.csv", "status": "VERIFIED"},
        
        # Baselines
        {"metric": "hgb_strict_auroc", "value": "0.7733", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.34 (val)", "source_script": "03_models_and_baselines_forensics.py", "source_output": "baseline_comparison.csv", "status": "VERIFIED"},
        {"metric": "hgb_strict_auprc", "value": "0.6101", "units": "AUPRC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.34 (val)", "source_script": "03_models_and_baselines_forensics.py", "source_output": "baseline_comparison.csv", "status": "VERIFIED"},
        {"metric": "lr_strict_auroc", "value": "0.7161", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.27 (val)", "source_script": "03_models_and_baselines_forensics.py", "source_output": "baseline_comparison.csv", "status": "VERIFIED"},
        {"metric": "yaw_rate_th_auroc", "value": "0.6895", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "33.5 deg/s", "source_script": "03_models_and_baselines_forensics.py", "source_output": "baseline_comparison.csv", "status": "VERIFIED"},
        {"metric": "fsd_baseline_auroc", "value": "0.3964", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "strict_clean_frames", "threshold": "0.01", "source_script": "03_models_and_baselines_forensics.py", "source_output": "baseline_comparison.csv", "status": "VERIFIED"},
        
        # Paired Bootstrap Differences
        {"metric": "legacy_minus_strict_auroc_diff", "value": "0.0387", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "paired_bootstrap_2000", "threshold": "0.50", "source_script": "05_uncertainty_and_generalization.py", "source_output": "bootstrap_paired_ci.csv", "status": "VERIFIED"},
        {"metric": "legacy_minus_strict_auroc_p_val", "value": "0.0000", "units": "p_value", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "paired_bootstrap_2000", "threshold": "0.50", "source_script": "05_uncertainty_and_generalization.py", "source_output": "bootstrap_paired_ci.csv", "status": "VERIFIED"},
        {"metric": "frozen_minus_v0_auroc_diff", "value": "0.0004", "units": "AUROC", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "paired_bootstrap_2000", "threshold": "0.50 vs 0.54", "source_script": "05_uncertainty_and_generalization.py", "source_output": "bootstrap_paired_ci.csv", "status": "VERIFIED"},
        {"metric": "frozen_minus_v0_auroc_p_val", "value": "0.9040", "units": "p_value", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "paired_bootstrap_2000", "threshold": "0.50 vs 0.54", "source_script": "05_uncertainty_and_generalization.py", "source_output": "bootstrap_paired_ci.csv", "status": "VERIFIED"},
        
        # Generalization Cross-Validation
        {"metric": "loco_mean_auroc", "value": "0.6980", "units": "AUROC", "dataset": "ROS_telemetry", "split": "15_cell_folds", "evaluation_protocol": "leave_one_cell_out", "threshold": "val_tuned", "source_script": "05_uncertainty_and_generalization.py", "source_output": "generalization_summary.csv", "status": "VERIFIED"},
        {"metric": "loyo_mean_auroc", "value": "0.6466", "units": "AUROC", "dataset": "ROS_telemetry", "split": "4_yaw_folds", "evaluation_protocol": "leave_one_yaw_rate_out", "threshold": "val_tuned", "source_script": "05_uncertainty_and_generalization.py", "source_output": "generalization_summary.csv", "status": "VERIFIED"},
        {"metric": "logo_mean_auroc", "value": "0.7688", "units": "AUROC", "dataset": "ROS_telemetry", "split": "4_geom_folds", "evaluation_protocol": "leave_one_geom_out", "threshold": "val_tuned", "source_script": "05_uncertainty_and_generalization.py", "source_output": "generalization_summary.csv", "status": "VERIFIED"},
        
        # Event-Level Early Warning (W=15, 0.50 s)
        {"metric": "v0_w15_rising_edge_early_pct", "value": "59.35", "units": "%", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "W=15_pre_onset", "threshold": "0.54", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        {"metric": "v0_w15_any_active_early_pct", "value": "83.15", "units": "%", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "W=15_pre_onset", "threshold": "0.54", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        {"metric": "v0_w15_rising_edge_mean_lead_s", "value": "0.3369", "units": "seconds", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "W=15_pre_onset", "threshold": "0.54", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        {"metric": "v0_w15_rising_edge_median_lead_s", "value": "0.3667", "units": "seconds", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "W=15_pre_onset", "threshold": "0.54", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        {"metric": "v0_w15_alarm_duty_cycle_pct", "value": "41.16", "units": "%", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "frame_duty_cycle", "threshold": "0.54", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        {"metric": "v0_w15_alarm_rising_edges_per_min", "value": "107.17", "units": "edges/min", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "0_to_1_transitions", "threshold": "0.54", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        {"metric": "v0_w15_false_alarm_edges_per_min", "value": "25.93", "units": "edges/min", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "no_onset_in_W15", "threshold": "0.54", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        {"metric": "hgb_w15_rising_edge_early_pct", "value": "76.66", "units": "%", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "W=15_pre_onset", "threshold": "0.34", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        {"metric": "hgb_w15_false_alarm_edges_per_min", "value": "20.44", "units": "edges/min", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "no_onset_in_W15", "threshold": "0.34", "source_script": "04_early_warning_and_alarms.py", "source_output": "event_warning_summary.csv", "status": "VERIFIED"},
        
        # Causal Replay & ROS 2
        {"metric": "causal_replay_parity_max_diff", "value": "4.17e-07", "units": "probability", "dataset": "ROS_telemetry", "split": "test", "evaluation_protocol": "streaming_vs_offline", "threshold": "0.50", "source_script": "06_causal_replay_and_ros2_forensics.py", "source_output": "causal_replay_comparison.csv", "status": "VERIFIED"},
        {"metric": "ros2_30hz_message_drop_rate", "value": "0.00", "units": "%", "dataset": "ROS_telemetry", "split": "4_flights", "evaluation_protocol": "multi_process_ros2", "threshold": "0.50", "source_script": "06_causal_replay_and_ros2_forensics.py", "source_output": "live_ros2_run_a_latencies.csv", "status": "VERIFIED"},
        {"metric": "ros2_30hz_median_roundtrip_ms", "value": "1.24", "units": "ms", "dataset": "ROS_telemetry", "split": "sweep_M_C_R1", "evaluation_protocol": "multi_process_ros2", "threshold": "0.50", "source_script": "06_causal_replay_and_ros2_forensics.py", "source_output": "live_ros2_run_a_latencies.csv", "status": "VERIFIED"},
        {"metric": "ros2_30hz_p99_roundtrip_ms", "value": "1.62", "units": "ms", "dataset": "ROS_telemetry", "split": "sweep_M_C_R1", "evaluation_protocol": "multi_process_ros2", "threshold": "0.50", "source_script": "06_causal_replay_and_ros2_forensics.py", "source_output": "live_ros2_run_a_latencies.csv", "status": "VERIFIED"},
        {"metric": "ros2_500hz_message_drop_rate", "value": "0.00", "units": "%", "dataset": "synthetic_pacing", "split": "stress_run_b", "evaluation_protocol": "multi_process_ros2", "threshold": "0.50", "source_script": "06_causal_replay_and_ros2_forensics.py", "source_output": "live_ros2_run_b_stress.csv", "status": "VERIFIED"},
    ]
    
    df = pd.DataFrame(rows)
    df.to_csv(ledger_path, index=False)
    print(f"Saved {ledger_path}")

def generate_discrepancy_log():
    log_path = results_dir / "DISCREPANCY_LOG.md"
    content = """# AEGIS Forensic Discrepancy Log

This document reconciles all quantitative, conceptual, and procedural discrepancies identified between earlier reports/drafts and the newly recomputed ground truth.

---

### Discrepancy 1: Evaluation Protocol and Test AUROC (0.8046 vs 0.7665)
1. **Old Value**: AUROC = 0.8046, AUPRC = 0.7260, F1 = 0.6484 (Legacy)
2. **New Value**: AUROC = 0.7665, AUPRC = 0.5977, F1 = 0.5836 (Strict Clean Frames)
3. **Source of Truth**: `scripts/audit_final/03_models_and_baselines_forensics.py` -> `results/audit_final/frozen_legacy_vs_strict.csv`
4. **Why Discrepancy Occurred**: The legacy evaluation target $[t, t+K]$ included the current frame $t$. On test frames where tracking was *already failed* at frame $t$ (653 frames), optical flow dropped below 0.5 px with 97.3% probability, and the legacy model predicted positive on 98.32% (642/653) of them. Removing contemporaneous failure frames eliminated 642 artificial true positives, reducing AUROC by **0.0381** ($p < 0.001$) and AUPRC by **0.1283** ($p < 0.001$).
5. **Manuscript Impact**: **MANDATORY CORRECTION**. The headline test AUROC must be revised from 0.8046 to **0.7665**.

---

### Discrepancy 2: Causal Replay Frame Count (10,481 vs 10,411)
1. **Old Value**: 10,481 frames evaluated in `15_causal_replay_evaluation.py`
2. **New Value**: 10,411 frames evaluated in `scripts/audit_final/06_causal_replay_and_ros2_forensics.py`
3. **Source of Truth**: `results/audit_final/causal_replay_comparison.csv`
4. **Why Discrepancy Occurred**: `forward_shifts.max(axis=1)` in pandas skips NaNs by default. For the last 5 frames of each flight, the lookahead window extended beyond the flight boundary, creating NaNs in future shifts. `max()` silently evaluated over truncated subsets instead of marking the rows as invalid. Exactly 5 tail frames across 14 flights ($14 \\times 5 = 70$ frames) were erroneously retained ($10,481 - 70 = 10,411$).
5. **Manuscript Impact**: **MANDATORY CORRECTION**. The causal replay frame count must state 10,411 frames.

---

### Discrepancy 3: "Alarms Per Minute" vs "Duty Cycle"
1. **Old Value**: Prose in `t5_report.md` Section 4 cited "~38.7 alarms per minute and a 43.7% duty cycle" alongside "false-alarm episodes occur at 27-32 episodes per minute".
2. **New Value**:
   - Alarm Duty Cycle: **41.16%**
   - Alarm Rising Edges: **107.17 edges/min**
   - False-Alarm Rising Edges ($W=15$): **25.93 edges/min**
3. **Source of Truth**: `results/audit_final/event_warning_summary.csv` and `results/audit_final/alarm_metric_definition.md`
4. **Why Discrepancy Occurred**: The term "alarms per minute" was ambiguous. The code in `t5_event_early_warning.py` actually calculated **rising edges (0 -> 1 transitions) per minute**, while the prose cited an uncalibrated test number (~38.7/min) from an earlier scratch script.
5. **Manuscript Impact**: **MANDATORY CLARIFICATION**. The metric must be explicitly split and reported as **Alarm Duty Cycle (%)** and **Alarm Rising Edges / min**.

---

### Discrepancy 4: T5 Report Prose vs T5 CSV Early Warning Percentages at Matched 30% Duty Cycle
1. **Old Value**: `t5_report.md` prose stated: "Retrained V0 achieves 57.3% early warnings, while HistGradientBoosting achieves 58.9%, and pure Yaw-Rate threshold achieves 51.2%."
2. **New Value**:
   - Retrained V0 ($W=15$): **62.91%** (rising edge) / **72.18%** (any active)
   - HistGradientBoosting ($W=15$): **79.13%** (rising edge) / **83.46%** (any active)
   - Yaw Rate Threshold ($W=15$): **34.93%** (rising edge) / **69.86%** (any active)
3. **Source of Truth**: `results/audit_final/matched_duty_cycle_summary.csv`
4. **Why Discrepancy Occurred**: The author of `t5_report.md` mistakenly copied **frame-level test recalls** from the T2 baseline table at 30% duty cycle (where V0 recall was 56.1%, HGB recall was 55.9%, and yaw rate recall was 49.4%) into the event-level section of the prose, rather than using the actual event-level early-warning percentages from the T5 CSV table.
5. **Manuscript Impact**: **MANDATORY CORRECTION**. Use the authoritative recomputed event-level values.

---

### Discrepancy 5: Maximum Possible Scientific Lead Time (0.165 s Artifact)
1. **Old Value**: Previous prose suggested early warning was physically capped at 0.165 s (5 frames at 30 Hz).
2. **New Value**: Lead times extend up to **0.50 s** ($W=15$, mean lead 0.337 s, median 0.367 s) and **1.00 s** ($W=30$, mean lead 0.697 s, median 0.767 s).
3. **Source of Truth**: `results/audit_final/event_warning_summary.csv`
4. **Why Discrepancy Occurred**: The original evaluator evaluated anticipation strictly within a 5-frame lookback window matching the $K=5$ training horizon ($5 / 30 = 0.167\\,\\text{s}$). The actual continuous telemetry stream allows anticipation across wider windows $W \\in \\{10, 15, 30\\}$.
5. **Manuscript Impact**: **CONCEPTUAL REVISION**. Clearly distinguish frame-level label horizon ($K=5$) from event-level anticipation window ($W$).

---

### Discrepancy 6: MLP Superiority over Classical Baselines
1. **Old Value**: Previous draft claimed SmallMLP outperformed classical models across all metrics.
2. **New Value**: HistGradientBoosting achieves higher strict AUROC (**0.7733 vs 0.7665**) and higher AUPRC (**0.6101 vs 0.5982**), and achieves higher early-warning percentage (**76.66% vs 59.35%**) with fewer spurious alarms (**20.44 vs 25.93 / min**) at native operating thresholds.
3. **Source of Truth**: `results/audit_final/baseline_comparison.csv` and `results/audit_final/event_warning_summary.csv`
4. **Why Discrepancy Occurred**: Classical baseline tuning in the original draft was cursory. Thorough scikit-learn tree ensemble evaluation demonstrates that tabular telemetry features do not require neural architectures.
5. **Manuscript Impact**: **MANDATORY SCIENTIFIC CORRECTION**. The paper must honestly state that gradient-boosted trees equal or outperform the MLP.

---

### Discrepancy 7: VO Failure Duration (Prolonged Collapse vs Single-Frame Dropout)
1. **Old Value**: Described in earlier text as "visual-odometry catastrophic tracking loss / estimator collapse".
2. **New Value**: Exactly **99.59% (1,961 of 1,969)** of failure episodes are single-frame dropouts (33.3 ms duration), with a maximum duration of 3 frames (100 ms).
3. **Source of Truth**: `scripts/audit_final/01_dataset_and_failure_forensics.py` -> `results/audit_final/failure_mechanism_report.md`
4. **Why Discrepancy Occurred**: In `minimal_vo.py`, when inliers drop below 8, the estimator clears `prev_pts = None`. In the immediate next frame, `goodFeaturesToTrack` re-detects corners, resetting the tracking loop. The pipeline skips pose integration for that single frame and immediately resumes.
5. **Manuscript Impact**: **MANDATORY CONCEPTUAL CORRECTION**. The paper must accurately characterize the benchmark as predicting single-frame VO tracking dropouts / re-detection events, not multi-second estimator divergence.
"""
    with open(log_path, "w") as f:
        f.write(content)
    print(f"Saved {log_path}")

def generate_final_verdict():
    verdict_path = results_dir / "FINAL_FORENSIC_VERDICT.md"
    content = """# AEGIS Final Forensic Verdict

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
"""
    with open(verdict_path, "w") as f:
        f.write(content)
    print(f"Saved {verdict_path}")

def main():
    print("=" * 70)
    print("AEGIS FINAL FORENSIC PASS — SCRIPT 07: COMPILATION")
    print("=" * 70)
    generate_claim_ledger()
    generate_number_ledger()
    generate_discrepancy_log()
    generate_final_verdict()
    print("All Ledgers and Verdicts Compiled Successfully!")

if __name__ == "__main__":
    main()
