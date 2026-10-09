# Forensic Report: Dataset & Visual Odometry Failure Mechanism

**Execution Timestamp**: October 2026  
**Dataset Location**: `<RAW_DATA_DIR>`  
**Analyzed Scope**: All 42 eligible flights across Train, Validation, and Test partitions.  

## 1. Dataset Integrity & Partition Summary

- **Total Eligible Flights**: 42
  - **Train**: 15 flights (11,328 active frames)
  - **Validation**: 13 flights (9,736 active frames)
  - **Test (Held-Out)**: 14 flights (10,551 active frames)
- **Flight Overlap**: **Zero overlap** (sets are strictly disjoint).
- **Total Raw Video Frames Recorded**: 51,041
- **Total In-Flight Active Frames ($pos_z \ge 2.0\,\text{m}$)**: 31,615
- **Total Failure Frames ($num\_inliers\_pose < 8$)**: 1,978 (6.26% base rate)

## 2. Failure Episode Dynamics

- **Total Failure Episodes**: 1,969
- **Single-Frame Episodes**: **1,961 / 1,969 (99.5937%)**
- **Episode Length Distribution**:
  - Length = 1 frame(s): 1,961 episodes (99.59%)
  - Length = 2 frame(s): 7 episodes (0.36%)
  - Length = 3 frame(s): 1 episodes (0.05%)
- **Maximum Episode Length**: **3 frames** ($pprox 0.10\,\text{s}$ at 30 Hz).
- **Inter-Failure Gap Statistics**:
  - Sample Count: 1,927
  - Median Gap: **10.0 frames** ($pprox 0.33\,\text{s}$)
  - Mean Gap: **14.75 frames** ($pprox 0.49\,\text{s}$)
  - Min / Max: 1 frame / 186 frames (0.03s to 6.14s)

## 3. Visual Odometry Implementation Audit (`minimal_vo.py`)

Direct inspection of `<ROS_REPO>/src/core/minimal_vo.py` establishes the exact causal control flow responsible for these single-frame dropouts:

```python
# minimal_vo.py line 258-264:
if self.prev_img is None or self.prev_pts is None or len(self.prev_pts) < 100:
    pts = cv2.goodFeaturesToTrack(cv_img, maxCorners=2000, qualityLevel=0.001, minDistance=5)
    self.prev_pts = pts
    self.prev_img = cv_img
    num_detected = len(pts) if pts is not None else 0
    return num_detected, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0

# minimal_vo.py line 347-348:
if num_matched >= 8:
    ... # compute Essential matrix and recover pose
else:
    self.prev_pts = None
```

### Control Flow Sequence during a Failure Frame:
1. **Tracking Breakdown**: When rapid rotational motion causes KLT optical flow matching to yield `num_matched < 8`, the estimator triggers line 348: `self.prev_pts = None`.
2. **Immediate Re-detection (Next Frame)**: In the subsequent frame, line 258 detects that `self.prev_pts is None`. The node immediately invokes OpenCV `goodFeaturesToTrack(cv_img, maxCorners=2000)`.
3. **Synchronous Return of Zeroes**: Because optical flow cannot be tracked across a keyframe re-initialization, line 263 explicitly returns:
   - `num_matched = 0`
   - `num_inliers_pose = 0` (which triggers the failure flag `num_inliers_pose < 8`)
   - `vel_mean = 0.0` (which zeroes out `feature_vel_mean`)
   - `rel_tx, rel_ty, rel_tz = 0.0, 0.0, 0.0`
4. **Pose Update Freeze**: Lines 218–224 show that the node publishes the frozen previous position `self.curr_pos` without updating the integration.
5. **Instantaneous Recovery**: In the following frame, `self.prev_pts` is now populated with $\approx 1000\text{--}2000$ fresh corner points. KLT tracking succeeds normally, matching hundreds of points and yielding $>400$ inliers.

### Definitive Classification:
The failure events in AEGIS are **NOT prolonged estimator breakdowns or progressive mechanical sensor failures**.
They are **algorithmic single-frame tracking dropouts followed by instantaneous feature re-detection**, during which the visual odometry pose update is skipped for exactly one 33.3 ms sample.

## 4. Per-Flight Characterization Table

| flight | split | family | raw_frames | active_frames | failure_frames | failure_rate_pct | episodes | single_frame_episodes | single_pct | max_episode_len |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sweep_A_B_R1 | train | A_B | 1213 | 736 | 63 | 8.5600 | 63 | 63 | 100.0000 | 1 |
| sweep_A_B_R2 | val | A_B | 1221 | 727 | 62 | 8.5300 | 61 | 60 | 98.3600 | 2 |
| sweep_A_B_R3 | test | A_B | 1224 | 745 | 57 | 7.6500 | 56 | 55 | 98.2100 | 2 |
| sweep_A_C_R1 | test | A_C | 1218 | 759 | 76 | 10.0100 | 76 | 76 | 100.0000 | 1 |
| sweep_A_C_R2 | train | A_C | 1216 | 724 | 59 | 8.1500 | 59 | 59 | 100.0000 | 1 |
| sweep_A_C_R3 | val | A_C | 1213 | 747 | 71 | 9.5000 | 70 | 69 | 98.5700 | 2 |
| sweep_A_H_R1 | train | A_H | 1212 | 752 | 36 | 4.7900 | 36 | 36 | 100.0000 | 1 |
| sweep_A_H_R2 | test | A_H | 1219 | 753 | 32 | 4.2500 | 32 | 32 | 100.0000 | 1 |
| sweep_A_H_R3 | val | A_H | 1223 | 766 | 41 | 5.3500 | 41 | 41 | 100.0000 | 1 |
| sweep_A_S_R1 | val | A_S | 1215 | 754 | 35 | 4.6400 | 35 | 35 | 100.0000 | 1 |
| sweep_A_S_R2 | train | A_S | 1222 | 752 | 33 | 4.3900 | 33 | 33 | 100.0000 | 1 |
| sweep_A_S_R3 | test | A_S | 1218 | 756 | 31 | 4.1000 | 31 | 31 | 100.0000 | 1 |
| sweep_E_B_R1 | val | E_B | 1213 | 763 | 65 | 8.5200 | 65 | 65 | 100.0000 | 1 |
| sweep_E_B_R2 | train | E_B | 1217 | 759 | 66 | 8.7000 | 66 | 66 | 100.0000 | 1 |
| sweep_E_B_R3 | test | E_B | 1217 | 750 | 59 | 7.8700 | 59 | 59 | 100.0000 | 1 |
| sweep_E_C_R1 | test | E_C | 1214 | 738 | 96 | 13.0100 | 95 | 94 | 98.9500 | 2 |
| sweep_E_C_R2 | train | E_C | 1223 | 755 | 97 | 12.8500 | 97 | 97 | 100.0000 | 1 |
| sweep_E_C_R3 | val | E_C | 1216 | 729 | 85 | 11.6600 | 85 | 85 | 100.0000 | 1 |
| sweep_E_H_R1 | train | E_H | 1213 | 746 | 59 | 7.9100 | 59 | 59 | 100.0000 | 1 |
| sweep_E_H_R2 | val | E_H | 1224 | 764 | 65 | 8.5100 | 65 | 65 | 100.0000 | 1 |
| sweep_E_H_R3 | test | E_H | 1217 | 750 | 62 | 8.2700 | 62 | 62 | 100.0000 | 1 |
| sweep_E_S_R1 | val | E_S | 1213 | 755 | 83 | 10.9900 | 83 | 83 | 100.0000 | 1 |
| sweep_E_S_R2 | test | E_S | 1211 | 737 | 79 | 10.7200 | 79 | 79 | 100.0000 | 1 |
| sweep_E_S_R3 | train | E_S | 1217 | 758 | 81 | 10.6900 | 81 | 81 | 100.0000 | 1 |
| sweep_G_B_R2 | test | G_B | 1215 | 751 | 17 | 2.2600 | 17 | 17 | 100.0000 | 1 |
| sweep_G_B_R3 | train | G_B | 1209 | 772 | 19 | 2.4600 | 19 | 19 | 100.0000 | 1 |
| sweep_G_C_R1 | test | G_C | 1210 | 745 | 22 | 2.9500 | 21 | 20 | 95.2400 | 2 |
| sweep_G_C_R2 | train | G_C | 1221 | 766 | 22 | 2.8700 | 22 | 22 | 100.0000 | 1 |
| sweep_G_C_R3 | val | G_C | 1181 | 803 | 21 | 2.6200 | 20 | 19 | 95.0000 | 2 |
| sweep_G_S_R1 | val | G_S | 1215 | 722 | 14 | 1.9400 | 14 | 14 | 100.0000 | 1 |
| sweep_G_S_R2 | test | G_S | 1210 | 826 | 25 | 3.0300 | 23 | 22 | 95.6500 | 3 |
| sweep_G_S_R3 | train | G_S | 1214 | 804 | 19 | 2.3600 | 19 | 19 | 100.0000 | 1 |
| sweep_M_B_R1 | val | M_B | 1218 | 714 | 33 | 4.6200 | 33 | 33 | 100.0000 | 1 |
| sweep_M_B_R2 | train | M_B | 1216 | 757 | 36 | 4.7600 | 36 | 36 | 100.0000 | 1 |
| sweep_M_B_R3 | test | M_B | 1215 | 750 | 41 | 5.4700 | 41 | 41 | 100.0000 | 1 |
| sweep_M_C_R1 | test | M_C | 1220 | 749 | 35 | 4.6700 | 34 | 33 | 97.0600 | 2 |
| sweep_M_C_R2 | train | M_C | 1208 | 755 | 35 | 4.6400 | 35 | 35 | 100.0000 | 1 |
| sweep_M_C_R3 | val | M_C | 1217 | 744 | 33 | 4.4400 | 33 | 33 | 100.0000 | 1 |
| sweep_M_H_R1 | train | M_H | 1215 | 749 | 21 | 2.8000 | 21 | 21 | 100.0000 | 1 |
| sweep_M_S_R1 | val | M_S | 1214 | 748 | 30 | 4.0100 | 30 | 30 | 100.0000 | 1 |
| sweep_M_S_R2 | test | M_S | 1218 | 742 | 32 | 4.3100 | 32 | 32 | 100.0000 | 1 |
| sweep_M_S_R3 | train | M_S | 1216 | 743 | 30 | 4.0400 | 30 | 30 | 100.0000 | 1 |
