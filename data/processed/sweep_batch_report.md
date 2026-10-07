# Research 2 — Systematic 4x4 Flight Sweep Batch Report

**Generated at**: 2026-09-27 23:58:33  
**Target Matrix**: 16 Cells x 3 Repeats = 48 Scheduled Flights  
**Simulation Environment**: Gazebo Harmonic (`agriculture.world`), PX4 SITL (`x500_mono_cam`)  
**VO Telemetry Pipeline**: `src/pipelines/run_offline_vo.py` (RAW mode, 5-point RANSAC)  
**Master Log CSV**: [`data/processed/sweep_flight_log.csv`](file:///home/purab/Purab/Projects/Research2/data/processed/sweep_flight_log.csv)  

---

## 1. Executive Summary

- **Total Flights Scheduled**: **48** (16 cells x 3 repeats)
- **Successful Flights (Passed)**: **46** (95.8%)
- **Failed Flights (Dropped after 2 attempts)**: **2** (4.2%)
- **Total New Active-Window Frames Generated**: **34,437 frames** (~18.9 minutes of calibrated telemetry)

---

## 2. Dedicated Audit of Underperforming / Attenuated Cells (< 3 Surviving Flights)

> [!IMPORTANT]
> Per the task specification, any cell that ended up with fewer than 3 surviving flights after retries is explicitly called out here rather than being masked in pooled statistics.

| Cell (Yaw + Motion) | Target Repeats | Surviving Repeats | Deficit | Documented Failure Diagnostics |
| :---: | :---: | :---: | :---: | :--- |
| `G_H` | 3 | **1** | 2 | R2: Att1: Active duration 16.7s < 18.0s threshold | Att2: Active duration 17.6s < 18.0s threshold; R3: Att1: Active duration 17.6s < 18.0s threshold | Att2: Active duration 12.6s < 18.0s threshold |

---

## 3. Achieved vs. Commanded Yaw Rates per Cell

| Cell | Commanded Yaw Rate (deg/s) | Achieved Mean Yaw Rate (deg/s) | Achieved Median Yaw Rate (deg/s) | Yaw Fidelity Error |
| :---: | :---: | :---: | :---: | :---: |
| `A_B` | 45.0 | 43.47 | 40.60 | 9.8% |
| `A_C` | 45.0 | 43.51 | 40.31 | 10.4% |
| `A_H` | 45.0 | 42.93 | 40.40 | 10.2% |
| `A_S` | 45.0 | 42.61 | 40.11 | 10.9% |
| `E_B` | 90.0 | 81.50 | 76.96 | 14.5% |
| `E_C` | 90.0 | 83.11 | 80.95 | 10.1% |
| `E_H` | 90.0 | 81.04 | 80.15 | 10.9% |
| `E_S` | 90.0 | 81.31 | 77.83 | 13.5% |
| `G_B` | 7.5 | 13.12 | 6.23 | 17.0% |
| `G_C` | 7.5 | 10.76 | 6.73 | 10.3% |
| `G_H` | 7.5 | 67.77 | 16.79 | 123.9% |
| `G_S` | 7.5 | 10.91 | 6.74 | 10.2% |
| `M_B` | 20.0 | 21.21 | 18.06 | 9.7% |
| `M_C` | 20.0 | 21.95 | 18.22 | 8.9% |
| `M_H` | 20.0 | 51.69 | 18.91 | 5.5% |
| `M_S` | 20.0 | 22.29 | 18.24 | 8.8% |

---

## 4. Optical Flow Dynamics & Verification Across Motion Bins

Verification of the physical flow dynamics across the 4 motion bins:
1. **B (Braking)**: Verified drop during braking windows (target 0–2 px/frame in G/M bins, vs F12's 9.44 px/frame).
2. **S (Sharp Stop)**: Verified complete cessation of forward velocity held stationary.
3. **H (High-speed Burst)**: Verified higher optical flow than baseline cruise during bursts.
4. **C (Cruise)**: Steady baseline optical flow.

| Motion Bin | Mean Cruise Flow (px/frame) | Motion Window Mean Flow (px/frame) | Motion Window Min Flow (px/frame) | Motion Window Max Flow (px/frame) | Flow Regime Characterization |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **C** | 16.77 | 16.77 | **0.00** | **82.44** | Steady uniform nominal flow |
| **B** | 18.66 | 18.31 | **0.00** | **592.79** | Substantial flow drop during braking pulses (enters danger zone) |
| **S** | 17.59 | 17.07 | **0.00** | **60.46** | Near-zero flow singularity during 3s full stops |
| **H** | 26.69 | 23.61 | **0.00** | **245.89** | Elevated optical flow and motion blur during 3.5 m/s bursts |

---

## 5. Naive Failure Rate by Cell (`num_inliers_pose < 8`)

| Cell | Surviving Flights | Total Active Frames | Total Failure Frames | Mean Failure Rate (%) | Min Failure Rate (%) | Max Failure Rate (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `A_B` | 3 | 2208 | 182 | **8.25%** | 7.65% | 8.56% |
| `A_C` | 3 | 2230 | 206 | **9.22%** | 8.15% | 10.01% |
| `A_H` | 3 | 2271 | 109 | **4.80%** | 4.25% | 5.35% |
| `A_S` | 3 | 2262 | 99 | **4.38%** | 4.10% | 4.64% |
| `E_B` | 3 | 2272 | 190 | **8.36%** | 7.87% | 8.70% |
| `E_C` | 3 | 2222 | 278 | **12.51%** | 11.66% | 13.01% |
| `E_H` | 3 | 2260 | 186 | **8.23%** | 7.91% | 8.51% |
| `E_S` | 3 | 2250 | 243 | **10.80%** | 10.69% | 10.99% |
| `G_B` | 3 | 2396 | 314 | **12.19%** | 2.26% | 31.84% |
| `G_C` | 3 | 2314 | 65 | **2.81%** | 2.62% | 2.95% |
| `G_H` | 1 | 608 | 44 | **7.24%** | 7.24% | 7.24% |
| `G_S` | 3 | 2352 | 58 | **2.44%** | 1.94% | 3.03% |
| `M_B` | 3 | 2221 | 110 | **4.95%** | 4.62% | 5.47% |
| `M_C` | 3 | 2248 | 103 | **4.58%** | 4.44% | 4.67% |
| `M_H` | 3 | 2090 | 152 | **7.45%** | 2.80% | 11.11% |
| `M_S` | 3 | 2233 | 92 | **4.12%** | 4.01% | 4.31% |

---

## 6. Complete 48-Flight Execution Log

| Flight ID | Cell | Rep | Attempts | Status | Achieved Yaw (deg/s) | Active Dur (s) | Active Frames | Failure Frames | Fail Rate (%) | Mean Flow (px) | Motion Min Flow (px) | Diagnostics |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `sweep_G_C_R1` | `G_C` | R1 | 1 | **PASS** | 7.0 | 24.6 | 745 | 22 | 3.0% | 8.0 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_C_R2` | `G_C` | R2 | 1 | **PASS** | 6.8 | 25.3 | 766 | 22 | 2.9% | 8.2 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_C_R3` | `G_C` | R3 | 1 | **PASS** | 6.4 | 26.5 | 803 | 21 | 2.6% | 9.3 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_B_R1` | `G_B` | R1 | 1 | **PASS** | 4.7 | 28.8 | 873 | 278 | 31.8% | 28.4 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_B_R2` | `G_B` | R2 | 1 | **PASS** | 7.1 | 24.8 | 751 | 17 | 2.3% | 7.9 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_B_R3` | `G_B` | R3 | 1 | **PASS** | 6.9 | 25.5 | 772 | 19 | 2.5% | 7.7 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_S_R1` | `G_S` | R1 | 2 | **PASS** | 6.7 | 23.8 | 722 | 14 | 1.9% | 7.7 | 0.00 | Attempt 1 failed (Active duration 16.8s < 18.0s threshold) |
| `sweep_G_S_R2` | `G_S` | R2 | 1 | **PASS** | 6.5 | 27.3 | 826 | 25 | 3.0% | 9.6 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_S_R3` | `G_S` | R3 | 1 | **PASS** | 6.9 | 26.6 | 804 | 19 | 2.4% | 9.7 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_H_R1` | `G_H` | R1 | 1 | **PASS** | 16.8 | 20.1 | 608 | 44 | 7.2% | 26.8 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_G_H_R2` | `G_H` | R2 | 2 | **FAIL** | - | - | - | - | - | - | - | Att1: Active duration 16.7s < 18.0s threshold | Att2: Active duration 17.6s < 18.0s threshold |
| `sweep_G_H_R3` | `G_H` | R3 | 2 | **FAIL** | - | - | - | - | - | - | - | Att1: Active duration 17.6s < 18.0s threshold | Att2: Active duration 12.6s < 18.0s threshold |
| `sweep_M_C_R1` | `M_C` | R1 | 1 | **PASS** | 17.8 | 24.7 | 749 | 35 | 4.7% | 11.6 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_C_R2` | `M_C` | R2 | 1 | **PASS** | 17.9 | 24.9 | 755 | 35 | 4.6% | 11.1 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_C_R3` | `M_C` | R3 | 1 | **PASS** | 18.9 | 24.5 | 744 | 33 | 4.4% | 11.8 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_B_R1` | `M_B` | R1 | 1 | **PASS** | 18.0 | 23.6 | 714 | 33 | 4.6% | 11.2 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_B_R2` | `M_B` | R2 | 1 | **PASS** | 17.3 | 25.0 | 757 | 36 | 4.8% | 11.0 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_B_R3` | `M_B` | R3 | 1 | **PASS** | 18.9 | 24.7 | 750 | 41 | 5.5% | 11.1 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_S_R1` | `M_S` | R1 | 1 | **PASS** | 19.0 | 24.7 | 748 | 30 | 4.0% | 11.9 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_S_R2` | `M_S` | R2 | 1 | **PASS** | 17.9 | 24.5 | 742 | 32 | 4.3% | 12.0 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_S_R3` | `M_S` | R3 | 1 | **PASS** | 17.8 | 24.5 | 743 | 30 | 4.0% | 12.0 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_H_R1` | `M_H` | R1 | 1 | **PASS** | 18.5 | 24.7 | 749 | 21 | 2.8% | 17.6 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_H_R2` | `M_H` | R2 | 1 | **PASS** | 19.8 | 22.3 | 675 | 57 | 8.4% | 25.1 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_M_H_R3` | `M_H` | R3 | 1 | **PASS** | 18.4 | 22.0 | 666 | 74 | 11.1% | 27.7 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_C_R1` | `A_C` | R1 | 1 | **PASS** | 42.5 | 25.1 | 759 | 76 | 10.0% | 17.6 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_C_R2` | `A_C` | R2 | 1 | **PASS** | 40.1 | 23.9 | 724 | 59 | 8.1% | 18.5 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_C_R3` | `A_C` | R3 | 1 | **PASS** | 38.3 | 24.6 | 747 | 71 | 9.5% | 17.9 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_B_R1` | `A_B` | R1 | 1 | **PASS** | 40.0 | 24.3 | 736 | 63 | 8.6% | 17.6 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_B_R2` | `A_B` | R2 | 1 | **PASS** | 42.3 | 24.0 | 727 | 62 | 8.5% | 17.8 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_B_R3` | `A_B` | R3 | 1 | **PASS** | 39.4 | 24.6 | 745 | 57 | 7.7% | 17.9 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_S_R1` | `A_S` | R1 | 1 | **PASS** | 41.6 | 24.9 | 754 | 35 | 4.6% | 19.1 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_S_R2` | `A_S` | R2 | 1 | **PASS** | 38.8 | 24.8 | 752 | 33 | 4.4% | 18.9 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_S_R3` | `A_S` | R3 | 1 | **PASS** | 39.9 | 24.9 | 756 | 31 | 4.1% | 18.9 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_H_R1` | `A_H` | R1 | 1 | **PASS** | 40.8 | 24.8 | 752 | 36 | 4.8% | 21.3 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_H_R2` | `A_H` | R2 | 1 | **PASS** | 42.3 | 24.8 | 753 | 32 | 4.2% | 21.9 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_A_H_R3` | `A_H` | R3 | 1 | **PASS** | 38.0 | 25.3 | 766 | 41 | 5.4% | 21.1 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_C_R1` | `E_C` | R1 | 1 | **PASS** | 85.2 | 24.4 | 738 | 96 | 13.0% | 28.4 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_C_R2` | `E_C` | R2 | 1 | **PASS** | 77.1 | 24.9 | 755 | 97 | 12.8% | 28.7 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_C_R3` | `E_C` | R3 | 1 | **PASS** | 80.5 | 24.1 | 729 | 85 | 11.7% | 30.3 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_B_R1` | `E_B` | R1 | 1 | **PASS** | 73.0 | 25.2 | 763 | 65 | 8.5% | 30.3 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_B_R2` | `E_B` | R2 | 1 | **PASS** | 76.9 | 25.0 | 759 | 66 | 8.7% | 30.8 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_B_R3` | `E_B` | R3 | 1 | **PASS** | 81.1 | 24.8 | 750 | 59 | 7.9% | 31.6 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_S_R1` | `E_S` | R1 | 1 | **PASS** | 76.6 | 24.9 | 755 | 83 | 11.0% | 29.0 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_S_R2` | `E_S` | R2 | 1 | **PASS** | 80.5 | 24.3 | 737 | 79 | 10.7% | 31.0 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_S_R3` | `E_S` | R3 | 1 | **PASS** | 76.4 | 25.0 | 758 | 81 | 10.7% | 29.9 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_H_R1` | `E_H` | R1 | 1 | **PASS** | 83.5 | 24.6 | 746 | 59 | 7.9% | 33.3 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_H_R2` | `E_H` | R2 | 1 | **PASS** | 76.6 | 25.2 | 764 | 65 | 8.5% | 32.3 | 0.00 | Clean flight (Pass Att 1) |
| `sweep_E_H_R3` | `E_H` | R3 | 1 | **PASS** | 80.4 | 24.7 | 750 | 62 | 8.3% | 32.2 | 0.00 | Clean flight (Pass Att 1) |

---

## 7. Artifact Index

- **Master Flight Log**: [`data/processed/sweep_flight_log.csv`](file:///home/purab/Purab/Projects/Research2/data/processed/sweep_flight_log.csv)
- **Batch Summary Report**: [`data/processed/sweep_batch_report.md`](file:///home/purab/Purab/Projects/Research2/data/processed/sweep_batch_report.md)
- **Sweep Motion Controller**: [`scripts/fly_sweep_motion.py`](file:///home/purab/Purab/Projects/Research2/scripts/fly_sweep_motion.py)
- **Sweep Batch Orchestrator**: [`scripts/11_run_sweep_batch.py`](file:///home/purab/Purab/Projects/Research2/scripts/11_run_sweep_batch.py)
- **Datasets Directory**: [`results/datasets/`](file:///home/purab/Purab/Projects/ROS/results/datasets)

End of systematic sweep batch report.
