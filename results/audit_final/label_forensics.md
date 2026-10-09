# Forensic Report: Label Definition & Contemporaneous Feature Leakage

## 1. Exact Label Implementations

In `scripts/12_build_expanded_dataset.py`, the legacy target was constructed as:

```python
# Legacy label construction in scripts/12_build_expanded_dataset.py:
forward_shifts = pd.concat([df_k['is_failure'].shift(-step) for step in range(k + 1)], axis=1)
df_k['is_failure_within_next_k'] = forward_shifts.max(axis=1)
```

Because `range(k + 1)` includes `step = 0`, this evaluates:
$$y_{\text{legacy}}(t) = \max(f_t, f_{t+1}, f_{t+2}, \dots, f_{t+K})$$

In contrast, a strict anticipatory warning label requires:
$$y_{\text{strict}}(t) = \max(f_{t+1}, f_{t+2}, \dots, f_{t+K})$$

On any frame where visual odometry is *already failing* ($f_t = 1$), the event is ongoing, not imminent.
In strict evaluation, all frames with $f_t = 1$ are excluded from the test set.
In strict training, all frames with $f_t = 1$ are excluded from training and validation to prevent the model from learning contemporaneous cues.

## 2. Contemporaneous Optical Flow Leakage

- Total active flight frames: **31,615**
- Total failure frames ($num\_inliers\_pose < 8$): **1,978**
- Total frames with $feature\_vel\_mean < 0.5\,\text{px}$: **1,983**
- Intersection: **1,925 frames**

- **$P(\text{failure} \mid \text{flow} < 0.5\,\text{px})$**: **97.08%**
- **$P(\text{flow} < 0.5\,\text{px} \mid \text{failure})$**: **97.32%**

### Optical Flow Magnitude Distributions:
- **Failure Frames ($num\_inliers < 8$)**:
  - Median: **0.0000 px/frame**
  - 75th Percentile: **0.0000 px/frame**
  - Mean: 0.2609 px/frame
- **Non-Failure Frames ($num\_inliers \ge 8$)**:
  - Median: **17.7847 px/frame**
  - Mean: 19.8917 px/frame

Because `feature_vel_mean_lag0` is included in the 15-feature input vector, any model with access to lag 0 has a near-perfect contemporaneous indicator of whether visual odometry has collapsed in the current frame.

## 3. Deterministic Identity of `is_r_frame`

- Tested condition: `is_r_frame == (eis_yaw_rate_deg > 15.0)`
- Total evaluated frames: **31,615**
- Mismatches observed: **0**
- Exact match rate: **100.0000%**

`is_r_frame` is **not an independent feature**. It is an exact, deterministic step function of body yaw rate.

## 4. Test Split Quarantine Accounting

- Total active frames in 14 held-out test flights: **10,551**
- Warmup frames dropped (first 5 per flight): **70**
- Tail frames dropped (last 5 per flight): **70**
- Legacy evaluated frames ($10,551 - 140$): **10,411**
- Ongoing failure frames in legacy evaluation ($f_t = 1$): **653**
- **Strict evaluated frames** ($10,411 - 653$): **9,758**
