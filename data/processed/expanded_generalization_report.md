# Research 2 — Expanded 42-Flight Generalization Evaluation Report

## 1. Executive Summary

This evaluation tests whether the telemetry-based VO failure predictor (Candidate B) learns a generalizable signal across a substantially broader 4x4 flight sweep distribution, or merely fit the original 33-flight trajectory dataset.

### Key Results Table
| Metric | Original 33-Flight Baseline | Expanded 42-Flight Sweep | Delta |
|---|---|---|---|
| **Eligible Flights** | 33 | 42 | +9 (+27.3%) |
| **Train Flights / Frames** | 11 / 7,215 | 15 / 11,178 | +4 / +3,963 |
| **Val Flights / Frames** | 11 / 7,155 | 13 / 9,606 | +2 / +2,451 |
| **Test Flights / Frames** | 11 / 7,248 | 14 / 10,411 | +3 / +3,163 |
| **Test Pos Rate** | 34.02% | 34.62% | +0.60% |
| **AUROC (K=5)** | **0.8234** | **0.8046** | **-0.0188** |
| **F1 Score (K=5)** | **0.6634** | **0.6484** | **-0.0150** |
| **Precision (K=5)** | 0.6033 | 0.5808 | -0.0225 |
| **Recall (K=5)** | 0.7368 | 0.7336 | -0.0032 |

### Prediction Horizon Sensitivity
| Horizon | Original AUROC | Original F1 | Expanded AUROC | Expanded F1 |
|---|---|---|---|---|
| **K=5 (~165 ms, Primary)** | 0.8234 | 0.6634 | 0.8046 | 0.6484 |
| **K=3 (~99 ms)** | 0.8037 | 0.5401 | 0.8052 | 0.5512 |
| **K=2 (~66 ms)** | 0.7816 | 0.4497 | 0.8174 | 0.4789 |

## 2. Per-Flight Held-Out Performance (K=5)

| Flight ID | Cell | Frames | Positives | Pos Rate | AUROC | AUPRC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| `sweep_A_B_R3` | A_B | 735 | 312 | 42.4% | 0.7168 | 0.7025 | 0.6161 | 0.5850 | 0.6506 |
| `sweep_A_C_R1` | A_C | 749 | 419 | 55.9% | 0.8016 | 0.8575 | 0.7270 | 0.7686 | 0.6897 |
| `sweep_A_H_R2` | A_H | 743 | 191 | 25.7% | 0.8393 | 0.6840 | 0.5758 | 0.4243 | 0.8953 |
| `sweep_A_S_R3` | A_S | 746 | 174 | 23.3% | 0.7710 | 0.6387 | 0.4841 | 0.3495 | 0.7874 |
| `sweep_E_B_R3` | E_B | 740 | 340 | 45.9% | 0.7675 | 0.7868 | 0.6578 | 0.5055 | 0.9412 |
| `sweep_E_C_R1` | E_C | 728 | 506 | 69.5% | 0.7331 | 0.8738 | 0.8147 | 0.7304 | 0.9209 |
| `sweep_E_H_R3` | E_H | 740 | 354 | 47.8% | 0.7549 | 0.7842 | 0.6769 | 0.5304 | 0.9350 |
| `sweep_E_S_R2` | E_S | 727 | 416 | 57.2% | 0.7903 | 0.8525 | 0.7579 | 0.6296 | 0.9519 |
| `sweep_G_B_R2` | G_B | 741 | 87 | 11.7% | 0.7842 | 0.5316 | 0.5493 | 0.7091 | 0.4483 |
| `sweep_G_C_R1` | G_C | 735 | 104 | 14.1% | 0.8280 | 0.5651 | 0.5424 | 0.6575 | 0.4615 |
| `sweep_G_S_R2` | G_S | 816 | 114 | 14.0% | 0.8022 | 0.4513 | 0.4128 | 0.4327 | 0.3947 |
| `sweep_M_B_R3` | M_B | 740 | 222 | 30.0% | 0.6847 | 0.5802 | 0.4375 | 0.7143 | 0.3153 |
| `sweep_M_C_R1` | M_C | 739 | 191 | 25.8% | 0.7059 | 0.6001 | 0.4982 | 0.7553 | 0.3717 |
| `sweep_M_S_R2` | M_S | 732 | 174 | 23.8% | 0.6690 | 0.5228 | 0.4462 | 0.6744 | 0.3333 |

## 3. Generalization Assessment

**Classification**: `STRONGER EVIDENCE OF GENERALIZATION`  

**Rationale**: Held-out AUROC (0.8046) and F1 (0.6484) remain substantially above chance across 14 completely unseen flights from 13 distinct sweep cells, with performance virtually identical to the original 33-flight baseline (AUROC gap: -0.0188). The model demonstrates genuine flight-level generalization across varying yaw rates and trajectory geometries.
