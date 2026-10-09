# AEGIS Forensic Discrepancy Log

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
4. **Why Discrepancy Occurred**: `forward_shifts.max(axis=1)` in pandas skips NaNs by default. For the last 5 frames of each flight, the lookahead window extended beyond the flight boundary, creating NaNs in future shifts. `max()` silently evaluated over truncated subsets instead of marking the rows as invalid. Exactly 5 tail frames across 14 flights ($14 \times 5 = 70$ frames) were erroneously retained ($10,481 - 70 = 10,411$).
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
4. **Why Discrepancy Occurred**: The original evaluator evaluated anticipation strictly within a 5-frame lookback window matching the $K=5$ training horizon ($5 / 30 = 0.167\,\text{s}$). The actual continuous telemetry stream allows anticipation across wider windows $W \in \{10, 15, 30\}$.
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
