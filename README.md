# Research 2 — Learned VO Failure Predictor

Working title: *IMU-trusted lightweight alternative to sliding-window
pure-rotation handling*

A direct, falsifiable follow-up to Research 1's VFO null result (linear/
single-variable correlation analysis found no predictive signal for
monocular VO tracking failure, all |r| < 0.16). This project tests
whether a *learned nonlinear* combination of the same telemetry features
finds signal where VFO's linear analysis found none.

Target: Nov 1, 2026 (stretch goal, zero risk to the ETH SSRF application,
which depends only on Research 1 — already complete, DOI #1/#2/#3 all
published).

## Relationship to Research 1 (VO_Research)

This is a **separate repo** from VO_Research on purpose — the tooling
here (ML training pipeline) is genuinely different from R1's ROS2/
Gazebo/PX4 simulation stack. But the **data is not duplicated** into this
repo. Instead, `scripts/01_fetch_data.py` pulls the specific CSVs needed
directly from your local VO_Research repo path, preserving the
provenance chain back to the DOI-pinned data (DOI #2:
`10.5281/zenodo.22951257`).

**Before running anything**, edit `configs/phase0_config.yaml` and set
`source_repo.local_path` to your actual VO_Research repo location (e.g.
`/home/purab/Purab/Projects/ROS`).

## Phase 0 — Data Assembly & Labeling

Phase 0's job is narrow and complete in itself: produce a clean,
validated, labeled, flight-split dataset. No feature engineering, no
modeling — that's Phase 1 and Phase 2.

### What's locked in this phase (see `configs/phase0_config.yaml` for full rationale)

- **Ground truth condition: RAW only.** DELAYED-TRI is excluded from
  labeling — its 44-58% "failure" rates in F6/F9/F10 are the confirmed
  zero-motion-fallback artifact from R1, not genuine tracking
  degradation. Training on those labels would teach the model to predict
  a pipeline bug.
- **HOVER family excluded entirely.** Its 45% failure rate reflects
  startup/baseline feature starvation at t≈0, a different phenomenon
  from the sustained rotation-induced failure this research studies.
- **Failure definition:** `num_inliers_pose < 8` — the pipeline's own
  documented pose-update threshold, same definition used throughout R1.
- **Active window:** `pos_z >= 2.0` — excludes takeoff/landing ground
  phases, consistent with R1.
- **Label type: single-frame** (is *this* frame a failure), not a
  forward-looking window. This keeps the model-vs-VFO comparison in
  Phase 3 apples-to-apples, since VFO's own analysis was per-frame.
  Forward-window prediction ("will it fail in the next K frames") is a
  planned extension for Phase 1/2, not baked into Phase 0.
- **Split: flight-level, not frame-level**, stratified by family. This
  is the single most important safeguard here — frames within one flight
  are correlated, so splitting at the frame level would leak information
  and produce an inflated, untrustworthy result.
- **3 GT heading-discontinuity flights are flagged, not excluded**
  (`p3_F6_L2_R1`, `p3_F6_L2_R3`, `p3_F10_L3_R2`) — wherever they land in
  the split, that should be reported alongside any downstream result.

### Expected numbers (from the pre-Phase-0 data audit)

- 33 flights (24 core + 9 exploratory, HOVER excluded)
- ~21,948 active-window frames
- ~1,953 genuine failure frames (~8.9% base rate)
- These are approximate — `02_build_labels.py` validates the actual
  computed numbers against these and flags anything more than 5% off for
  review (small deviations are expected/fine; large ones mean something
  changed and needs a look before treating the dataset as final).

### Running it

```bash
pip install -r requirements.txt

# Edit configs/phase0_config.yaml first — set source_repo.local_path

# Run the whole pipeline:
bash scripts/run_phase0.sh

# Or run steps individually:
python scripts/01_fetch_data.py     # pulls raw_vo.csv + dataset_gt.csv per flight
python scripts/02_build_labels.py   # builds labeled frame table + validation report
python scripts/03_split_flights.py  # locks the flight-level train/val/test split
```

### Outputs (gitignored — local working data, not committed)

- `data/raw/` — cached CSVs pulled from VO_Research, plus
  `_fetch_manifest.txt` recording exactly what was pulled from where
- `data/processed/frames_labeled.csv` — one row per active-window frame,
  with `is_failure`, `family`, `run_dir`, `gt_discontinuity_flag`
- `data/processed/flight_manifest.csv` — one row per flight (33 rows)
- `data/processed/flight_split.csv` — flight manifest + assigned
  train/val/test split (**this is the locked split — do not regenerate
  with a different seed to "improve" results**)
- `data/processed/phase0_validation_report.md` — the numbers, checked
  against audit expectations, per-family breakdown, and where the
  GT-discontinuity flights landed

### Gate before Phase 1

Read `data/processed/phase0_validation_report.md`. If all checks pass
(or deviations are small and understood), Phase 0 is done. Do not start
feature engineering or modeling until this report has been reviewed.

## What's NOT in this repo yet

- Feature engineering (Phase 1)
- Baseline model / LSTM training (Phase 2)
- Held-out evaluation (Phase 3)
- Any new-flight data collection scripts (Phase 4, only if needed after
  Phase 3's verdict)

These will be added as later deliverables, same phase-gated discipline
as Research 1.
