# Phase 1 Feature Report — Research 2

Ground-truth condition: `RAW`  
Failure rule: `num_inliers_pose < 8`  
Active-window rule: `pos_z >= 2.0`  

## v1 Feature Set

| # | Column | Source | Notes |
|---|---|---|---|
| 1 | `eis_yaw_rate_deg` | `raw_vo.csv` | |omega_z|, deg/s |
| 2 | `feature_vel_mean` | `raw_vo.csv` | Mean KLT flow, px/frame. **Conflated in RAW mode** (rotation + translation mixed) |
| 3 | `is_r_frame` | `raw_vo.csv` | Binary: 1 if eis_yaw_rate_deg > 15.0 deg/s |
| 4 | `vel_residual` | Engineered | feature_vel_mean minus (intercept + slope x eis_yaw_rate_deg), fit on train only |

> **Explicitly absent:** `eis_gate_scale` -- always 1.0 in RAW condition, zero discriminative information.

## vel_residual Regression Fit

Model: `feature_vel_mean ~ intercept + slope x eis_yaw_rate_deg`  
Fit on: **11 train-split flights** (7,325 active-window frames)  

| Parameter | Value |
|---|---|
| intercept | 7.984742 px/frame |
| slope | 0.089102 px/frame per deg/s |
| R2 (train) | 0.059079 |

Interpretation: rotation (eis_yaw_rate_deg) explains **5.9%** of the variance in feature_vel_mean on the training set. `vel_residual` is the unexplained remainder -- a crude proxy for translational apparent motion not accounted for by yaw rate alone. This is not a precise flow decomposition: scene depth, feature distribution, and non-yaw rotations all affect the relationship in ways a global linear model cannot capture.

## v1 Feature Summary Statistics

Active-window frames only. All 33 flights (train + val + test).

### `eis_yaw_rate_deg`
| Stat | overall | is_failure=0 | is_failure=1 |
|---|---|---|---|
| n | 21948 | 19995 | 1953 |
| mean | 10.650189 | 10.850511 | 8.599277 |
| std | 29.594042 | 30.139627 | 23.191055 |
| min | 0.0 | 0.0 | 0.0006 |
| p25 | 0.0351 | 0.0347 | 0.0394 |
| p50 | 0.1084 | 0.1109 | 0.0875 |
| p75 | 3.367675 | 4.0693 | 0.6025 |
| max | 871.537 | 871.537 | 324.3833 |

### `feature_vel_mean`
| Stat | overall | is_failure=0 | is_failure=1 |
|---|---|---|---|
| n | 21948 | 19995 | 1953 |
| mean | 8.876117 | 9.695558 | 0.486604 |
| std | 13.345872 | 13.344822 | 10.059453 |
| min | 0.0 | 0.0241 | 0.0 |
| p25 | 2.03955 | 2.69695 | 0.0 |
| p50 | 5.15 | 5.621 | 0.0 |
| p75 | 10.273225 | 11.51405 | 0.0 |
| max | 472.4502 | 472.4502 | 437.7748 |

### `is_r_frame`
| Stat | overall | is_failure=0 | is_failure=1 |
|---|---|---|---|
| n | 21948 | 19995 | 1953 |
| mean | 0.198241 | 0.2019 | 0.160778 |
| std | 0.398684 | 0.401428 | 0.36742 |
| min | 0.0 | 0.0 | 0.0 |
| p25 | 0.0 | 0.0 | 0.0 |
| p50 | 0.0 | 0.0 | 0.0 |
| p75 | 0.0 | 0.0 | 0.0 |
| max | 1.0 | 1.0 | 1.0 |

### `vel_residual`
| Stat | overall | is_failure=0 | is_failure=1 |
|---|---|---|---|
| n | 21948 | 19995 | 1953 |
| mean | -0.057575 | 0.744017 | -8.264348 |
| std | 12.803029 | 12.745219 | 10.254859 |
| min | -74.009192 | -74.009192 | -36.887854 |
| p25 | -6.055514 | -5.465122 | -8.03308 |
| p50 | -3.015947 | -2.543223 | -7.991532 |
| p75 | 1.262273 | 2.013224 | -7.98648 |
| max | 464.458757 | 464.458757 | 429.745275 |

## Label Distribution

| | Frames | Fraction |
|---|---|---|
| is_failure=0 | 19,995 | 0.9110 |
| is_failure=1 | 1,953 | 0.0890 |
| **Total** | **21,948** | |

## Other Columns in frames_labeled.csv -- Inspection Only

These columns are present in `frames_labeled.csv` (carried over from `raw_vo.csv`) but are **not** part of the v1 feature set. Reported here for information only; no decision to include them has been made. Columns marked [CONST/ZERO] are constant or zero in RAW mode and carry no information.

| Column | dtype | n_unique | const/zero | min | max | mean | std | mean(fail=0) | mean(fail=1) |
|---|---|---|---|---|---|---|---|---|---|
| `bd_ratio` | float64 | 14874 | no | 5e-05 | 0.55615 | 0.01593 | 0.02518 | 0.01628 | 0.01228 |
| `consecutive_low_inliers` | int64 | 12 | no | 0.0 | 11.0 | 0.12147 | 0.54972 | 0.0 | 1.36508 |
| `eis_crop_pct` [CONST/ZERO] | float64 | 1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `eis_cum_warp_deg` [CONST/ZERO] | float64 | 1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `eis_dt_ms` [CONST/ZERO] | float64 | 1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `eis_warp_deg` [CONST/ZERO] | float64 | 1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `failure_triggered_streak` [CONST/ZERO] | bool | 1 | yes | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| `failure_triggered_window` [CONST/ZERO] | bool | 1 | yes | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| `feature_survival_rate` | float64 | 1373 | no | 0.0 | 1.0 | 0.90589 | 0.26084 | 0.97857 | 0.16176 |
| `feature_vel_max` | float64 | 19825 | no | 0.0 | 1502.3717 | 44.42871 | 131.25489 | 48.27125 | 5.08837 |
| `inlier_ratio` | float64 | 4503 | no | 0.0 | 1.0 | 0.80106 | 0.33768 | 0.8791 | 0.002 |
| `mean_lk_err` | float64 | 13152 | no | 0.0 | 45.4254 | 1.64806 | 1.89354 | 1.78306 | 0.266 |
| `num_active` | int64 | 1964 | no | 0.0 | 2000.0 | 827.19022 | 631.21821 | 757.72578 | 1538.37378 |
| `num_detected` | int64 | 1908 | no | 0.0 | 2000.0 | 839.77233 | 637.2268 | 771.35659 | 1540.21915 |
| `num_inliers` | int64 | 1942 | no | 0.0 | 2000.0 | 581.29852 | 510.35987 | 638.04621 | 0.31029 |
| `num_inliers_E` | int64 | 1955 | no | 0.0 | 2000.0 | 692.45772 | 575.42281 | 748.3924 | 119.79314 |
| `num_inliers_H` [CONST/ZERO] | int64 | 1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `num_inliers_pose` | int64 | 1942 | no | 0.0 | 2000.0 | 581.29852 | 510.35987 | 638.04621 | 0.31029 |
| `num_matched` | int64 | 1937 | no | 0.0 | 2000.0 | 700.98815 | 584.63535 | 757.72578 | 120.10292 |
| `num_pending` [CONST/ZERO] | int64 | 1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `promotions_this_frame` [CONST/ZERO] | int64 | 1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `rel_rot_deg` | float64 | 732 | no | 0.0 | 142.83 | 0.89783 | 2.25015 | 0.98553 | 0.0 |
| `rel_tx` | float64 | 19996 | no | -0.99995 | 1.0 | 0.22412 | 0.61852 | 0.24602 | 0.0 |
| `rel_ty` | float64 | 19996 | no | -1.0 | 0.99991 | -0.03742 | 0.44959 | -0.04108 | 0.0 |
| `rel_tz` | float64 | 19996 | no | -0.99999 | 1.0 | -0.0617 | 0.5205 | -0.06773 | 0.0 |
| `window_low_inlier_pct` | float64 | 41 | no | 0.0 | 65.57 | 9.049 | 10.29992 | 7.96728 | 20.12377 |

> **Note on RAW-constant columns:** `eis_crop_pct`, `eis_warp_deg`, `eis_cum_warp_deg` are 0.0 in RAW mode (EIS derotation never applied). `num_pending` and `promotions_this_frame` are 0 because delayed triangulation is not active in RAW mode. These cannot carry predictive signal within the RAW-only dataset.
