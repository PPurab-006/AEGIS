#!/usr/bin/env python3
"""
scripts/validate_gates_on_existing.py

Dry-run validation tool to apply verification gates 3(a) and 3(b)
to all existing flight directories under results/datasets (or data/raw).

Gates evaluated:
  - Duration gate: active window (pos_z >= 2.0) duration >= 18.0s
  - Envelope gate 3(a): active window local X in [-20, 20], local Y in [-25, 25], Alt in [0.5, 5.0]
  - PX4 failsafe gate 3(b): px4_sitl.log contains 'Failsafe activated', 'Failsafe: blind land', or 'invalid setpoints'
  - Motion script gate 3(c): reported as 'n/a (no motion.log)' for historical data

Excludes any *_attempt<N> archive directories from flight enumeration.
"""

import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import constants from 11_run_sweep_batch / fly_sweep_motion
try:
    from scripts.fly_sweep_motion import LOCAL_X_MIN, LOCAL_X_MAX, LOCAL_Y_MIN, LOCAL_Y_MAX
except ImportError:
    try:
        from fly_sweep_motion import LOCAL_X_MIN, LOCAL_X_MAX, LOCAL_Y_MIN, LOCAL_Y_MAX
    except ImportError:
        LOCAL_X_MIN, LOCAL_X_MAX = -20.0, 20.0
        LOCAL_Y_MIN, LOCAL_Y_MAX = -25.0, 25.0

SPAWN_X0 = 14.0505
SPAWN_Y0 = -7.5229
ENVELOPE_ALT_MIN = 0.5
ENVELOPE_ALT_MAX = 5.0

FAILSAFE_TARGETS = [
    "Failsafe activated",
    "Failsafe: blind land",
    "invalid setpoints",
]


def find_dataset_dir() -> Path:
    candidates = [
        Path("/home/purab/Purab/Projects/ROS/results/datasets"),
        REPO_ROOT / "data" / "raw",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("sweep_*")):
            return c
    raise RuntimeError("Could not find directory containing sweep_* datasets")


def load_flight_log_attempts() -> dict:
    log_csv = REPO_ROOT / "data" / "processed" / "sweep_flight_log.csv"
    attempts = {}
    if log_csv.exists():
        try:
            df = pd.read_csv(log_csv)
            for _, row in df.iterrows():
                fn = row.get("flight_name")
                att = row.get("attempts_used")
                if pd.notna(fn) and pd.notna(att):
                    attempts[str(fn)] = int(att)
        except Exception:
            pass
    return attempts


def evaluate_flight(flight_dir: Path, attempt_num: int) -> dict:
    result = {
        "flight": flight_dir.name,
        "attempt": attempt_num,
        "duration_gate": "ERROR",
        "envelope_gate": "ERROR",
        "failsafe_gate": "ERROR",
        "motion_gate": "n/a (no motion.log)",
        "first_envelope_viol": "-",
        "failsafe_lines": "-",
    }

    gt_path = flight_dir / "dataset_gt.csv"
    if not gt_path.exists():
        result["duration_gate"] = "FAIL (no GT)"
        result["envelope_gate"] = "FAIL (no GT)"
        return result

    try:
        df_gt = pd.read_csv(gt_path)
    except Exception as e:
        result["duration_gate"] = f"FAIL (read err: {e})"
        result["envelope_gate"] = f"FAIL (read err: {e})"
        return result

    t = df_gt["timestamp_total_sec"].values.astype(float)
    z = df_gt["pos_z"].values.astype(float)
    act_idx = np.where(z >= 2.0)[0]

    # Duration gate
    if len(act_idx) < 10:
        result["duration_gate"] = "FAIL (never reached 2.0m)"
        dur = 0.0
    else:
        dur = float(t[act_idx[-1]] - t[act_idx[0]])
        if dur >= 18.0:
            result["duration_gate"] = f"PASS ({dur:.1f}s)"
        else:
            result["duration_gate"] = f"FAIL ({dur:.1f}s < 18s)"

    # Envelope gate 3(a)
    if len(act_idx) >= 1:
        x_local = df_gt["pos_x"].values[act_idx] - SPAWN_X0
        y_local = df_gt["pos_y"].values[act_idx] - SPAWN_Y0
        z_act = z[act_idx]
        t_act = t[act_idx]

        out_of_bounds = (
            (x_local < LOCAL_X_MIN) | (x_local > LOCAL_X_MAX) |
            (y_local < LOCAL_Y_MIN) | (y_local > LOCAL_Y_MAX) |
            (z_act < ENVELOPE_ALT_MIN) | (z_act > ENVELOPE_ALT_MAX)
        )
        if np.any(out_of_bounds):
            first_idx = int(np.where(out_of_bounds)[0][0])
            t_v = t_act[first_idx]
            x_v = x_local[first_idx]
            y_v = y_local[first_idx]
            z_v = z_act[first_idx]
            result["envelope_gate"] = "FAIL"
            result["first_envelope_viol"] = f"t={t_v:.2f}s, X={x_v:.2f}, Y={y_v:.2f}, Z={z_v:.2f}"
        else:
            result["envelope_gate"] = "PASS"
            result["first_envelope_viol"] = "-"
    else:
        result["envelope_gate"] = "FAIL (no active samples)"

    # PX4 failsafe gate 3(b)
    px4_log = flight_dir / "px4_sitl.log"
    if px4_log.exists():
        found = []
        try:
            with open(px4_log, "r", errors="replace") as f:
                for line in f:
                    for target in FAILSAFE_TARGETS:
                        if target in line:
                            found.append(line.strip())
                            break
        except Exception as e:
            result["failsafe_gate"] = f"ERROR ({e})"

        if found:
            result["failsafe_gate"] = "FAIL"
            result["failsafe_lines"] = "; ".join(found)
        else:
            result["failsafe_gate"] = "PASS"
            result["failsafe_lines"] = "-"
    else:
        result["failsafe_gate"] = "PASS (no log)"

    return result


def main():
    dataset_dir = find_dataset_dir()
    attempts_map = load_flight_log_attempts()

    # Enumerate all sweep_* directories, strictly EXCLUDING *_attempt<N>
    all_entries = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sweep_")]
    valid_flight_dirs = [d for d in all_entries if not re.search(r"_attempt\d+$", d.name)]

    # Standard sorting by Yaw (G, M, A, E), Motion (C, B, S, H), Repeat (1, 2, 3)
    yaw_order = {"G": 0, "M": 1, "A": 2, "E": 3}
    motion_order = {"C": 0, "B": 1, "S": 2, "H": 3}

    def sort_key(d: Path):
        m = re.match(r"sweep_([GMAE])_([CBSH])_R(\d+)", d.name)
        if m:
            y, mo, r = m.groups()
            return (yaw_order.get(y, 9), motion_order.get(mo, 9), int(r), d.name)
        return (99, 99, 99, d.name)

    valid_flight_dirs.sort(key=sort_key)

    results = []
    for d in valid_flight_dirs:
        attempt_num = attempts_map.get(d.name, 1)
        res = evaluate_flight(d, attempt_num)
        results.append(res)

    # Format Markdown Table
    headers = [
        "flight", "attempt", "duration gate", "envelope gate", "failsafe gate",
        "motion gate", "first envelope violation (time, local X/Y/Z)", "failsafe lines found"
    ]
    keys = [
        "flight", "attempt", "duration_gate", "envelope_gate", "failsafe_gate",
        "motion_gate", "first_envelope_viol", "failsafe_lines"
    ]
    col_widths = [max(len(h), max((len(str(r[k])) for r in results), default=0)) for h, k in zip(headers, keys)]

    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_row = "-|-".join("-" * w for w in col_widths)

    print(header_row)
    print(sep_row)
    for r in results:
        vals = [
            str(r["flight"]).ljust(col_widths[0]),
            str(r["attempt"]).center(col_widths[1]),
            str(r["duration_gate"]).ljust(col_widths[2]),
            str(r["envelope_gate"]).ljust(col_widths[3]),
            str(r["failsafe_gate"]).ljust(col_widths[4]),
            str(r["motion_gate"]).ljust(col_widths[5]),
            str(r["first_envelope_viol"]).ljust(col_widths[6]),
            str(r["failsafe_lines"]).ljust(col_widths[7]),
        ]
        print(" | ".join(vals))

    # Summary
    n_total = len(results)
    n_dur_pass = sum(1 for r in results if r["duration_gate"].startswith("PASS"))
    n_env_pass = sum(1 for r in results if r["envelope_gate"] == "PASS")
    n_fs_pass  = sum(1 for r in results if r["failsafe_gate"].startswith("PASS"))

    print("\nSummary:")
    print(f"  Total flights evaluated: {n_total}")
    print(f"  Duration Gate PASS: {n_dur_pass}/{n_total} (Fails: {[r['flight'] for r in results if not r['duration_gate'].startswith('PASS')]})")
    print(f"  Envelope Gate PASS: {n_env_pass}/{n_total} (Fails: {[r['flight'] for r in results if r['envelope_gate'] != 'PASS']})")
    print(f"  PX4 Failsafe PASS:  {n_fs_pass}/{n_total} (Fails: {[r['flight'] for r in results if not r['failsafe_gate'].startswith('PASS')]})")


if __name__ == "__main__":
    main()
