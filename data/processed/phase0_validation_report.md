# Phase 0 Validation Report

Failure rule: `num_inliers_pose < 8`  

Active window rule: `pos_z >= 2.0`  

Ground truth condition: `RAW`  

Excluded: `DELAYED-TRI` (DELAYED-TRI's 44-58% "failure" rates in F6/F9/F10 are the confirmed zero-motion-...)  


## Checks

| Metric | Actual | Expected (audit) | Diff % | Status |
|---|---|---|---|---|
| n_flights | 33 | 33 | 0.0% | OK |
| active_frames_total | 21948 | 21948 | 0.0% | OK |
| active_failure_frames | 1953 | 1953 | 0.0% | OK |
| active_failure_rate_pct | 8.9 | 8.9 | 0.0% | OK |

## Per-family breakdown

| Family | Flights | Frames | Failures | Rate % |
|---|---|---|---|---|
| F1 | 3 | 2052 | 384 | 18.71 |
| F10 | 3 | 2050 | 119 | 5.8 |
| F11 | 3 | 1829 | 55 | 3.01 |
| F2 | 3 | 1967 | 66 | 3.36 |
| F3 | 3 | 1733 | 246 | 14.2 |
| F4 | 3 | 1873 | 166 | 8.86 |
| F5 | 3 | 2135 | 202 | 9.46 |
| F6 | 3 | 1953 | 171 | 8.76 |
| F7 | 3 | 2119 | 205 | 9.67 |
| F8 | 3 | 2124 | 192 | 9.04 |
| F9 | 3 | 2113 | 147 | 6.96 |

## GT-discontinuity flights (flagged, not excluded)

- `p3_F6_L2_R1` (F6): 55 failures / 674 frames
- `p3_F6_L2_R3` (F6): 60 failures / 629 frames
- `p3_F10_L3_R2` (F10): 45 failures / 692 frames
