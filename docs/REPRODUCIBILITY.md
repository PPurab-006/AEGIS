# AEGIS Replication & Reproducibility Guide

This document provides exhaustive instructions, environment specifications, artifact checksums, and execution steps to reproduce the forensic audit, experiments, tables, and figures of the AEGIS research preprint:  
**"AEGIS: Telemetry-Based Early Warning of Monocular Visual Odometry Tracking Dropouts During Aggressive UAV Flight"**.

---

## 1. Environment & Dependency Setup

### System Requirements
- **Operating System**: Linux x86_64 (tested on Ubuntu 26.04.1 LTS / 24.04 LTS / 22.04 LTS)
- **Python Version**: Python 3.14.4 (compatible with Python >= 3.10)
- **Middleware (Optional for live ROS 2 node reproduction)**: ROS 2

### Setting Up the Virtual Environment
```bash
# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Pinned dependencies include:
- `torch==2.14.0`
- `scikit-learn==1.9.1`
- `numpy==2.5.3`
- `pandas==3.0.6`
- `opencv-python==4.10.0.84`
- `joblib==1.6.0`
- `matplotlib==3.10.7`
- `scipy==1.18.1`
- `PyYAML==6.0.3`
- `pytest==9.0.2`

---

## 2. Model Weights & Artifact Checksums

Legacy filenames (`best_model.pth`, `retrained_v0_strict.pth`, `scaler.pkl`) have been superseded by verified model artifacts:

### 1. Retrained V0 Model (Strict Task)
- **Model Weights**: `models/audit_final/v0_strict_seed42.pt`  
  - **SHA-256**: `bbbd9ce5020d9d46ae59da22dcf82bfae2efd09d7f98f0e482fbf24dec921f43`
- **Standard Scaler**: `models/audit_final/v0_strict_scaler.joblib`  
  - **SHA-256**: `b444e21481c5e28ae5308a4b659ca5b7e6c01d82b124e0c405241fdc051282a7`
- **Performance**: Strict AUROC = 0.7665, AUPRC = 0.5982, F1 = 0.5775 (at $\theta^*=0.54$) / 0.5758 (at $\theta=0.50$).

### 2. Frozen Legacy Model (Evaluated under Strict Task)
- **Model Weights**: `models/expanded_mlp.pt`  
  - **SHA-256**: `93acd1048a6954a143aac873a3fa8a4548c8a6230aa574aa1ea3fcce17b78e9f`
- **Standard Scaler**: `models/expanded_scaler.joblib`  
  - **SHA-256**: `4e9c6ebacb5ac4a554ed69a56d4c7c48df8ef1036b53907181b6dc221550b105`
- **Performance**: Legacy protocol weights scored under the strict evaluation task: AUROC = 0.7665, AUPRC = 0.5977, F1 = 0.5836 (at $\theta=0.50$).

To verify checksums locally:
```bash
sha256sum models/audit_final/v0_strict_seed42.pt \
          models/audit_final/v0_strict_scaler.joblib \
          models/expanded_mlp.pt \
          models/expanded_scaler.joblib
```

---

## 3. Repository Inputs & Dataset Structure

### Included in Repository
- `data/processed/telemetry_frames.csv.gz`: Active flight telemetry across all 42 eligible flights (31,615 active frames).
- `data/processed/audit_strict_frames_k5.csv.gz`: Feature matrix with 15-dimensional lagged features ($K=5$ horizon) constructed strictly without future leakage.
- `data/processed/expanded_flight_split.csv`: Authoritative flight-level split manifest (15 train, 13 val, 14 test flights).
- `data/processed/expanded_eligible_flights.txt`: List of 42 eligible flight names after physical envelope gate filtering.

### Raw Camera Frames
Raw camera image frames (~12 GB of uncompressed camera images across 42 flights) are excluded from the git repository to keep the distribution package lightweight. To run scripts that read directly from raw flight datasets:
```bash
export AEGIS_RAW_DATA_DIR=/path/to/raw/datasets
# or
export AEGIS_DATA_DIR=/path/to/raw/datasets
```
If unset, scripts default to `<REPO_ROOT>/data/raw`.

---

## 4. Deterministic Seeds & Randomness

All training and evaluation pipelines lock exact random seeds:
- **Flight Partition Split**: Seed `42`
- **Retrained V0 (`SmallMLP`) Training**: Seed `42`
- **HistGradientBoosting Baseline**: Seed `42`
- **Logistic Regression Baseline**: Seed `42`
- **Bootstrap Confidence Intervals**: `np.random.RandomState(42)` (2,000 cluster resamples)
- **Multi-Seed Uncertainty Evaluation**: Seeds `0, 1, 2, 3, 4`

---

## 5. Exact Commands to Reproduce Audit & Results

Run the complete 7-stage forensic audit pipeline in order:
```bash
python3 scripts/audit_final/01_dataset_and_failure_forensics.py
python3 scripts/audit_final/02_label_and_leakage_forensics.py
python3 scripts/audit_final/03_models_and_baselines_forensics.py
python3 scripts/audit_final/04_early_warning_and_alarms.py
python3 scripts/audit_final/05_uncertainty_and_generalization.py
python3 scripts/audit_final/06_causal_replay_and_ros2_forensics.py
python3 scripts/audit_final/07_synthesis_and_claim_matrix.py
```

Run independent standalone verification and unit tests:
```bash
# Independent recomputation of core dataset, model, and causal replay statistics
python3 scripts/audit_final/verify_independent.py

# Automated unit tests ensuring zero causal future-information leakage
pytest tests/test_causal_leakage.py
```

### Reproducing Publication Figures
To re-generate all publication vector figures from the audited result tables:
```bash
python3 scripts/generate_preprint_figures.py
```
Output figures are rendered to `results/figures/preprint/`.
