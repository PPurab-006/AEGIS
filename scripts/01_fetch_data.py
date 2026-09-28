#!/usr/bin/env python3
"""
Phase 0, Step 1 — Fetch data from the source VO_Research (R1) repo.

This script does NOT duplicate R1's data into this repo permanently.
It copies only the specific CSVs needed (raw_vo.csv, dataset_gt.csv) into
this repo's data/raw/ directory, which is gitignored — a local working
cache, not a committed copy. This preserves provenance: the data always
traces back to the source repo path / commit, and re-running this script
is how you refresh the cache if source data ever changes.

Usage:
    python scripts/01_fetch_data.py --config configs/phase0_config.yaml
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_source_git_info(source_path: Path) -> str:
    """Best-effort: report the source repo's current commit for provenance
    logging. Never fatal if this fails (e.g. not a git repo, git not
    installed) — this is a nice-to-have breadcrumb, not a hard requirement.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN (could not read source repo git state)"


def fetch_flight(source_datasets_dir: Path, dest_raw_dir: Path, run_dir: str) -> dict:
    """Copy the files Phase 0 actually needs for one flight recording.

    Only copies raw_vo.csv (RAW condition ground truth) and dataset_gt.csv
    (for active-window filtering). Does NOT copy images/ or the other
    condition CSVs (eis_gated_vo.csv, gated_dt_def_a_vo.csv) — those
    aren't needed for Phase 0 labeling and copying image sequences would
    be a large, unnecessary duplication.
    """
    src = source_datasets_dir / run_dir
    dst = dest_raw_dir / run_dir
    dst.mkdir(parents=True, exist_ok=True)

    needed_files = ["raw_vo.csv", "dataset_gt.csv"]
    status = {"run_dir": run_dir, "ok": True, "missing": []}

    if not src.is_dir():
        status["ok"] = False
        status["missing"] = needed_files
        status["error"] = f"Source directory not found: {src}"
        return status

    for fname in needed_files:
        src_file = src / fname
        if not src_file.exists():
            status["ok"] = False
            status["missing"].append(fname)
            continue
        shutil.copy2(src_file, dst / fname)

    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "phase0_config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source_path = Path(config["source_repo"]["local_path"]).expanduser()
    datasets_subdir = config["source_repo"]["datasets_subdir"]
    source_datasets_dir = source_path / datasets_subdir

    repo_root = Path(__file__).resolve().parent.parent
    dest_raw_dir = repo_root / "data" / "raw"
    dest_raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source repo path : {source_path}")
    if not source_path.is_dir():
        print(
            f"\nERROR: source repo path does not exist: {source_path}\n"
            f"Edit configs/phase0_config.yaml -> source_repo.local_path "
            f"to point at your actual VO_Research repo location, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    git_info = get_source_git_info(source_path)
    print(f"Source repo HEAD : {git_info}")
    print(f"Fetching into    : {dest_raw_dir}\n")

    run_directories = config["run_directories"]
    all_statuses = []
    for family, run_dirs in run_directories.items():
        for run_dir in run_dirs:
            status = fetch_flight(source_datasets_dir, dest_raw_dir, run_dir)
            status["family"] = family
            all_statuses.append(status)
            marker = "OK" if status["ok"] else "MISSING"
            print(f"  [{marker:7s}] {family:6s} {run_dir}")

    n_total = len(all_statuses)
    n_ok = sum(1 for s in all_statuses if s["ok"])
    n_failed = n_total - n_ok

    print(f"\nFetched {n_ok}/{n_total} flights successfully.")

    if n_failed > 0:
        print(f"\n{n_failed} flight(s) had missing files:", file=sys.stderr)
        for s in all_statuses:
            if not s["ok"]:
                print(f"  - {s['run_dir']}: missing {s['missing']}", file=sys.stderr)
        print(
            "\nThis may mean run_directories in phase0_config.yaml doesn't "
            "match your actual repo layout, or the source path is wrong. "
            "Fix before proceeding to labeling.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Write a small provenance manifest alongside the cached data
    manifest_path = dest_raw_dir / "_fetch_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"source_repo_path: {source_path}\n")
        f.write(f"source_repo_git_head: {git_info}\n")
        f.write(f"n_flights_fetched: {n_ok}\n")
        f.write("flights:\n")
        for s in all_statuses:
            f.write(f"  - {s['family']}: {s['run_dir']}\n")
    print(f"\nProvenance manifest written to {manifest_path}")
    print("\nDone. Next: python scripts/02_build_labels.py")


if __name__ == "__main__":
    main()
