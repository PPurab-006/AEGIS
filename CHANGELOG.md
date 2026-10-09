# Changelog

All notable changes to the AEGIS project are documented in this file.

## [v1.1.0] - 2026-10-10

### Added
- **Strict-Label Forensic Audit Suite**: Added complete 7-stage forensic audit pipeline in `scripts/audit_final/01..07` evaluating telemetry failure prediction under strict future-only labels.
- **Audited Results & Evidence**: Added verified result tables and ledgers in `results/audit_final/` (`FINAL_NUMBER_LEDGER.csv`, `primary_metrics_audit.csv`, `baseline_comparison.csv`, `feature_ablation_comparison.csv`, `generalization_summary.csv`, `event_early_warning_summary.csv`, `PRE_RELEASE_VERIFICATION.md`, `setup_facts.json`).
- **Verified Strict Model Artifacts**: Added `models/audit_final/v0_strict_seed42.pt` (SHA-256 `bbbd9ce5...`) and `models/audit_final/v0_strict_scaler.joblib` (SHA-256 `b444e214...`).
- **Standalone Independent Verification**: Added `scripts/audit_final/verify_independent.py` recomputing core metrics directly from telemetry frames.
- **Automated Causal Leakage Tests**: Added unit tests in `tests/test_causal_leakage.py`.
- **Release & Archival Metadata**: Added `LICENSE` (Apache-2.0), `LICENSE-DATA.md` (CC BY 4.0), `CITATION.cff` (cff-version 1.2.0), and `.zenodo.json` (Zenodo concept DOI `10.5281/zenodo.23267668`).

### Changed
- **Documentation & Verified Setup Facts**: Corrected camera resolution to 1280×960, body-aligned pose 0 0 0 (0° pitch relative to body), horizontal FOV 1.74 rad, and 30 FPS. Corrected commanded yaw bins (Gentle 7.5°/s, Moderate 20.0°/s, Aggressive 45.0°/s, Extreme 90.0°/s) and motion dynamic profiles (Cruise, Braking, Sharp stop, High-speed burst). Documented simulation versions (PX4 tree v1.18.0-beta1, gz sim 10.5.0, ROS 2).
- **Reproducibility Documentation**: Overhauled `docs/REPRODUCIBILITY.md` with verified model paths, sha256 checksums, and execution instructions.
- **Pinned Dependencies**: Pinned exact package versions in `requirements.txt` for Python 3.14.4 (`torch==2.14.0`, `scikit-learn==1.9.1`, `numpy==2.5.3`, `pandas==3.0.6`, `opencv-python==4.10.0.84`, `joblib==1.6.0`, `matplotlib==3.10.7`, `pytest==9.0.2`, `PyYAML==6.0.3`, `scipy==1.18.1`).
- **Path Portability**: Stripped all absolute local paths across scripts, configs, and reports, replacing them with repo-relative paths and environment variable fallbacks (`AEGIS_RAW_DATA_DIR`, `AEGIS_DATA_DIR`, `AEGIS_PX4_DIR`, `AEGIS_ROS_DIR`).

### Removed
- Untracked legacy distribution archives (`.zip`) and superseded manuscript drafts (`docs/AEGIS_Preprint.pdf`, `docs/AEGIS_Preprint.docx`) from repository tracking.

---

## [v1.0.0] - 2026-09-28

- Initial release and proof of concept of learned telemetry-based Visual Odometry failure prediction.
