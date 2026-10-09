# Forensic Verification: Causal Replay vs Offline Evaluation

## 1. Frame Count Reconciliation

- **Total Legacy Replay Evaluated Frames**: **10,481** frames
- **Total Corrected Replay Evaluated Frames**: **10,411** frames
- **Total Offline Test Evaluated Frames**: **10,411** frames
- **Difference (Legacy vs Corrected)**: **+70** frames (exactly 5 frames per flight across 14 flights = 70 frames)

## 2. Root Cause of Previous Discrepancy

In the original legacy replay script (`scripts/15_causal_replay_evaluation.py`), the lookahead label was calculated using:
```python
forward_shifts = pd.concat([df_out['actual_is_failure'].shift(-step) for step in range(6)], axis=1)
df_out['k5_label'] = forward_shifts.max(axis=1)
```
By default, `pandas.DataFrame.max(axis=1)` executes with `skipna=True`. Consequently, the last 5 frames of each flight—where future shifts are NaN—were not assigned NaN. Instead, `max()` evaluated over the truncated subset of available future frames. As a result, the last 5 boundary frames were retained in the evaluation set (10,481 frames total).

In contrast, the offline training and evaluation pipeline explicitly dropped frames whose forward lookahead window exceeded the flight boundary (leaving exactly 10,411 frames).

When explicit NaN masking is applied via:
```python
tail_nan_mask = forward_shifts.isna().any(axis=1)
df_out.loc[tail_nan_mask, 'k5_label'] = np.nan
```
the causal replay evaluated frame count becomes **exactly 10,411 frames**, perfectly matching offline test data.

## 3. Streaming Numerical Parity with Offline Predictions

- **Maximum absolute difference between streaming predictor and offline PyTorch model**: `4.17e-07`
- **Mean absolute difference**: `3.23e-08`

Streaming feature ring buffer extraction, StandardScaler normalization, and MLP inference produce **numerically identical outputs down to floating-point machine precision (< 1e-6)** compared to batch offline evaluation.

## 4. Per-Flight Breakdown

| flight | raw_active_frames | legacy_replay_frames | corrected_replay_frames | offline_test_frames | tail_frames_discarded | max_prob_diff_vs_offline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sweep_A_B_R3 | 745 | 740 | 735 | 735 | 5 | 2.384185791015625e-07 |
| sweep_A_C_R1 | 759 | 754 | 749 | 749 | 5 | 3.5762786865234375e-07 |
| sweep_A_H_R2 | 753 | 748 | 743 | 743 | 5 | 3.5762786865234375e-07 |
| sweep_A_S_R3 | 756 | 751 | 746 | 746 | 5 | 2.980232238769531e-07 |
| sweep_E_B_R3 | 750 | 745 | 740 | 740 | 5 | 2.086162567138672e-07 |
| sweep_E_C_R1 | 738 | 733 | 728 | 728 | 5 | 4.172325134277344e-07 |
| sweep_E_H_R3 | 750 | 745 | 740 | 740 | 5 | 3.5762786865234375e-07 |
| sweep_E_S_R2 | 737 | 732 | 727 | 727 | 5 | 1.7881393432617188e-07 |
| sweep_G_B_R2 | 751 | 746 | 741 | 741 | 5 | 2.682209014892578e-07 |
| sweep_G_C_R1 | 745 | 740 | 735 | 735 | 5 | 3.5762786865234375e-07 |
| sweep_G_S_R2 | 826 | 821 | 816 | 816 | 5 | 3.5762786865234375e-07 |
| sweep_M_B_R3 | 750 | 745 | 740 | 740 | 5 | 2.980232238769531e-07 |
| sweep_M_C_R1 | 749 | 744 | 739 | 739 | 5 | 2.980232238769531e-07 |
| sweep_M_S_R2 | 742 | 737 | 732 | 732 | 5 | 2.384185791015625e-07 |