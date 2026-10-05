#!/usr/bin/env python3
"""
scripts/validate_gates_on_existing.py

Dry-run validation tool to apply verification gates to all existing
flight directories under results/datasets (or data/raw).

Gates evaluated:
  - Duration gate: active window (pos_z >= 2.0) duration >= 18.0s
  - Envelope gate 3(a): active window local X in [-20, 20], local Y in [-25, 25], Alt in [0.5, 5.0]
  - PX4 failsafe gate 3(b): px4_sitl.log contains 'Failsafe activated', 'Failsafe: blind land', or 'invalid setpoints'
  - Motion script gate 3(c): reported as 'n/a (no motion.log)' for historical data
  - Health/failsafe warning gate: parse attempt's ULog (with px4_sitl.log fallback) for:
      "Compass 0 fault", "Compass needs calibration", "Imbalanced propeller detected",
      "Attitude failure", any "Preflight Fail:" post-arm, "invalid setpoints",
      "Failsafe: blind land", "Failsafe activated" between arming and disarm/end.
      Pre-arm "Preflight Fail: system power unavailable" recorded as note only.
      Records first tilt > 45 deg for failed attempts.

Excludes any *_attempt<N> archive directories from flight enumeration.
"""

import os
import re
import sys
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy

try:
    from pyulog import ULog
except ImportError:
    ULog = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    batch_module = importlib.import_module("11_run_sweep_batch")
    compute_tilt_metrics = batch_module.compute_tilt_metrics
    classify_warning_timing = batch_module.classify_warning_timing
    compute_loss_of_control = batch_module.compute_loss_of_control
    determine_landing_script_initiated = batch_module.determine_landing_script_initiated
    compute_duration_gate = batch_module.compute_duration_gate
except Exception:
    compute_tilt_metrics = None
    classify_warning_timing = None
    compute_loss_of_control = None
    determine_landing_script_initiated = None
    compute_duration_gate = None

PX4_DIR = os.environ.get("PX4_DIR", str(Path.home() / "PX4-Autopilot"))

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

HEALTH_WARN_TARGETS = [
    "Compass 0 fault",
    "Compass needs calibration",
    "Imbalanced propeller detected",
    "Attitude failure",
    "invalid setpoints",
    "Failsafe: blind land",
    "Failsafe activated",
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


def find_ulog_path(flight_dir: Path) -> Path | None:
    px4_log = flight_dir / "px4_sitl.log"
    if not px4_log.exists():
        return None
    try:
        text = px4_log.read_text(errors="replace")
    except Exception:
        return None
    m = re.search(r"Opened full log file:\s*(\S+?\.ulg)", text)
    if not m:
        return None
    rel = m.group(1).strip().lstrip("./")
    px4_rootfs = Path(PX4_DIR) / "build" / "px4_sitl_default" / "rootfs"
    cands = [
        px4_rootfs / rel,
        Path(PX4_DIR) / rel,
        flight_dir / rel,
        flight_dir / Path(rel).name,
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def evaluate_flight(flight_dir: Path, attempt_num: int) -> dict:
    result = {
        "flight": flight_dir.name,
        "attempt": attempt_num,
        "duration_gate": "ERROR",
        "envelope_gate": "ERROR",
        "failsafe_gate": "ERROR",
        "motion_gate": "n/a (no motion.log)",
        "warn_gate": "ERROR",
        "warnings_or_notes": "-",
        "first_tilt_exceed_45_s": "-",
        "first_tilt_z_m": "-",
        "max_tilt_inflight_deg": "-",
        "max_tilt_motion_deg": "-",
        "max_tilt_motion22_deg": "-",
        "t_return_start_s": "-",
        "landing_command_s": "-",
        "max_tilt_post_landcmd_deg": "-",
        "z_at_max_tilt_post_landcmd": "-",
        "first_loss_of_control_s": "-",
        "loss_of_control_indicator": "-",
        "t_onset_tilt": "-",
        "t_onset_vz": "-",
        "t_onset_yaw": "-",
        "first_warn_time": "-",
        "warning_timing_class": "no_warning",
        "headline_eligible": False,
        "old_max_tilt_z2": "-",
        "first_tilt_any_s": "-",
        "first_envelope_viol": "-",
        "failsafe_lines": "-",
    }

    gt_path = flight_dir / "dataset_gt.csv"
    df_gt = None
    if not gt_path.exists():
        result["duration_gate"] = "FAIL (no GT)"
        result["envelope_gate"] = "FAIL (no GT)"
    else:
        try:
            df_gt = pd.read_csv(gt_path)
        except Exception as e:
            result["duration_gate"] = f"FAIL (read err: {e})"
            result["envelope_gate"] = f"FAIL (read err: {e})"

    act_idx = []
    if df_gt is not None:
        t = df_gt["timestamp_total_sec"].values.astype(float)
        z = df_gt["pos_z"].values.astype(float)
        act_idx = np.where(z >= 2.0)[0]

        # Duration gate
        if compute_duration_gate is not None:
            dur, dur_status, dur_pass = compute_duration_gate(df_gt, threshold_s=18.0)
            result["duration_gate"] = dur_status
        else:
            if len(act_idx) < 10:
                result["duration_gate"] = "FAIL (never reached 2.0m)"
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

    # Health and Failsafe Warnings Gate
    ulog_path = find_ulog_path(flight_dir)
    ulog_obj = None
    if ulog_path is not None and ULog is not None:
        try:
            ulog_obj = ULog(str(ulog_path))
        except Exception:
            ulog_obj = None

    all_warns = []
    post_arm_warns = []
    pre_notes = []
    t_arm = None
    t_disarm = None
    t_land_cmd = None

    if ulog_obj is not None:
        for msg in ulog_obj.logged_messages:
            text = msg.message
            ts = msg.timestamp / 1e6
            if "Armed by external command" in text and t_arm is None:
                t_arm = ts
            elif "Disarmed" in text and t_disarm is None:
                t_disarm = ts

        # Determine landing command time (earliest post-arm)
        for msg in ulog_obj.logged_messages:
            ts = msg.timestamp / 1e6
            text_l = msg.message.strip().lower()
            if t_arm is not None and ts < t_arm:
                continue
            is_land = (
                "landing at current position" in text_l or
                ("failsafe" in text_l and "land" in text_l) or
                "disarm" in text_l
            )
            if is_land:
                if t_land_cmd is None or ts < t_land_cmd:
                    t_land_cmd = ts

        # Check pre-arm notes from px4_sitl.log
        if px4_log.exists():
            try:
                for line in px4_log.read_text(errors="replace").splitlines():
                    if "Armed by external command" in line:
                        break
                    if "Preflight Fail: system power unavailable" in line:
                        if "Preflight Fail: system power unavailable" not in pre_notes:
                            pre_notes.append("Preflight Fail: system power unavailable")
                        if not any("system power unavailable" in w[1] for w in all_warns):
                            all_warns.insert(0, (0.0, "Preflight Fail: system power unavailable"))
            except Exception:
                pass

        for msg in ulog_obj.logged_messages:
            ts = msg.timestamp / 1e6
            text = msg.message.strip()

            matched = False
            for w in HEALTH_WARN_TARGETS:
                if w in text:
                    matched = True
                    break
            if not matched and "Preflight Fail:" in text:
                matched = True

            if matched:
                all_warns.append((ts, text))
                if t_arm is not None and ts >= t_arm and (t_disarm is None or ts <= t_disarm):
                    post_arm_warns.append((ts, text))
                elif t_arm is None or ts < t_arm:
                    if "system power unavailable" in text:
                        if text not in pre_notes:
                            pre_notes.append(text)

        if post_arm_warns:
            result["warn_gate"] = "FAIL"
            warn_strs = [f"{ts:.2f}s: {txt}" for ts, txt in post_arm_warns]
            result["warnings_or_notes"] = "; ".join(warn_strs)
        else:
            result["warn_gate"] = "PASS"
            result["warnings_or_notes"] = f"[Note: {'; '.join(pre_notes)}]" if pre_notes else "-"

    else:
        # Fallback to px4_sitl.log matching
        if px4_log.exists():
            after_arm = False
            try:
                for line in px4_log.read_text(errors="replace").splitlines():
                    if "Armed by external command" in line:
                        after_arm = True
                        continue
                    if not after_arm:
                        if "Preflight Fail: system power unavailable" in line:
                            if "Preflight Fail: system power unavailable" not in pre_notes:
                                pre_notes.append("Preflight Fail: system power unavailable")
                            if not any("system power unavailable" in w[1] for w in all_warns):
                                all_warns.insert(0, (0.0, "Preflight Fail: system power unavailable"))
                    else:
                        matched = False
                        for w in HEALTH_WARN_TARGETS:
                            if w in line:
                                matched = True
                                break
                        if not matched and "Preflight Fail:" in line:
                            matched = True
                        if matched:
                            all_warns.append((t_arm if t_arm is not None else 0.0, line.strip()))
                            post_arm_warns.append((None, line.strip()))
            except Exception:
                pass

            if post_arm_warns:
                result["warn_gate"] = "FAIL"
                result["warnings_or_notes"] = "; ".join(f"fallback: {txt}" for _, txt in post_arm_warns)
            else:
                result["warn_gate"] = "unverifiable-fallback"
                result["warnings_or_notes"] = f"[Note: {'; '.join(pre_notes)}]" if pre_notes else "-"
        else:
            result["warn_gate"] = "unverifiable-fallback"
            result["warnings_or_notes"] = "-"

    # Evaluate tilt and loss of control metrics whenever GT is available
    if df_gt is not None:
        if compute_tilt_metrics is not None:
            m_yaw = re.match(r"sweep_([GMAE])_", flight_dir.name)
            yaw_bin = m_yaw.group(1) if m_yaw else "G"
            lsi = True
            if determine_landing_script_initiated is not None:
                lsi = determine_landing_script_initiated(
                    flight_dir=flight_dir,
                    t_arm=t_arm,
                    t_land_cmd=t_land_cmd,
                    warnings=all_warns,
                )
            result["landing_script_initiated"] = lsi
            tm = compute_tilt_metrics(
                df_gt,
                t_arm=t_arm,
                t_land_cmd=t_land_cmd,
                t_disarm=t_disarm,
                yaw_bin=yaw_bin,
                landing_script_initiated=lsi,
            )
            result["first_tilt_exceed_45_s"] = str(tm["first_tilt_exceed_45_s"]) if tm["first_tilt_exceed_45_s"] != "" else "-"
            result["first_tilt_z_m"] = str(tm["first_tilt_z_m"]) if tm["first_tilt_z_m"] != "" else "-"
            result["max_tilt_inflight_deg"] = str(tm["max_tilt_inflight_deg"]) if tm["max_tilt_inflight_deg"] != "" else "-"
            result["max_tilt_motion_deg"] = str(tm.get("max_tilt_motion_deg", "")) if tm.get("max_tilt_motion_deg", "") != "" else "-"
            result["max_tilt_motion22_deg"] = str(tm.get("max_tilt_motion22_deg", "")) if tm.get("max_tilt_motion22_deg", "") != "" else "-"
            result["t_return_start_s"] = str(tm.get("t_return_start_s", "")) if tm.get("t_return_start_s", "") != "" else "-"
            result["landing_command_s"] = str(tm["landing_command_s"]) if tm["landing_command_s"] != "" else "-"
            result["landing_script_initiated"] = tm.get("landing_script_initiated", lsi)
            result["first_tilt_any_s"] = str(tm["first_tilt_any_s"]) if tm["first_tilt_any_s"] != "" else "-"
            result["max_tilt_post_landcmd_deg"] = str(tm.get("max_tilt_post_landcmd_deg", "")) if tm.get("max_tilt_post_landcmd_deg", "") != "" else "-"
            result["z_at_max_tilt_post_landcmd"] = str(tm.get("z_at_max_tilt_post_landcmd", "")) if tm.get("z_at_max_tilt_post_landcmd", "") != "" else "-"
            result["first_loss_of_control_s"] = str(tm.get("first_loss_of_control_s", "")) if tm.get("first_loss_of_control_s", "") != "" else "-"
            result["loss_of_control_indicator"] = str(tm.get("loss_of_control_indicator", "")) if tm.get("loss_of_control_indicator", "") != "" else "-"
            result["t_onset_tilt"] = str(tm.get("t_onset_tilt", "")) if tm.get("t_onset_tilt", "") != "" else "-"
            result["t_onset_vz"] = str(tm.get("t_onset_vz", "")) if tm.get("t_onset_vz", "") != "" else "-"
            result["t_onset_yaw"] = str(tm.get("t_onset_yaw", "")) if tm.get("t_onset_yaw", "") != "" else "-"

        # Old z >= 2.0 max tilt
        z_vals = df_gt["pos_z"].values.astype(float)
        act_mask = z_vals >= 2.0
        if np.any(act_mask):
            quats = df_gt[["rot_x", "rot_y", "rot_z", "rot_w"]].values
            rots = R_scipy.from_quat(quats)
            body_z_world = rots.apply([0, 0, 1])
            z_comp = np.clip(body_z_world[:, 2], -1.0, 1.0)
            tilt_deg = np.rad2deg(np.arccos(z_comp))
            result["old_max_tilt_z2"] = f"{np.max(tilt_deg[act_mask]):.2f}"
        else:
            result["old_max_tilt_z2"] = "-"

    # Warning timing classification (D3)
    first_loc_s = float(result["first_loss_of_control_s"]) if result["first_loss_of_control_s"] != "-" else None
    if classify_warning_timing is not None:
        result["warning_timing_class"] = classify_warning_timing(t_arm, first_loc_s, all_warns)
    if post_arm_warns and post_arm_warns[0][0] is not None:
        result["first_warn_time"] = f"{post_arm_warns[0][0]:.3f}"

    # Headline eligibility (D1): hardened gates PASS (pre-arm power note ignored) AND max_tilt_motion_deg < 45.0
    hardened_pass = (
        result["duration_gate"].startswith("PASS") and
        result["envelope_gate"] == "PASS" and
        result["failsafe_gate"].startswith("PASS") and
        result["warn_gate"] == "PASS"
    )
    max_motion_str = result["max_tilt_motion_deg"]
    tilt_ok = False
    if max_motion_str != "-":
        try:
            tilt_ok = float(max_motion_str) < 45.0
        except (ValueError, TypeError):
            tilt_ok = False
    result["headline_eligible"] = hardened_pass and tilt_ok

    result["pre_arm_notes"] = "; ".join(pre_notes)
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

    # Format Validator Markdown Table
    headers = [
        "flight", "att", "script_init", "max_motion", "max_mot22", "max_inflight", "first_tilt_exceed_45_s", "z_cross",
        "ret_start_s", "land_cmd_s", "max_tilt_post", "z_max_post", "old_tilt_z2", "first_any_s",
        "dur_gate", "env_gate", "failsafe", "warn_gate", "headline_eligible"
    ]
    keys = [
        "flight", "attempt", "landing_script_initiated", "max_tilt_motion_deg", "max_tilt_motion22_deg", "max_tilt_inflight_deg", "first_tilt_exceed_45_s", "first_tilt_z_m",
        "t_return_start_s", "landing_command_s", "max_tilt_post_landcmd_deg", "z_at_max_tilt_post_landcmd", "old_max_tilt_z2", "first_tilt_any_s",
        "duration_gate", "envelope_gate", "failsafe_gate", "warn_gate", "headline_eligible"
    ]
    col_widths = [max(len(h), max((len(str(r[k])) for r in results), default=0)) for h, k in zip(headers, keys)]

    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_row = "-|-".join("-" * w for w in col_widths)

    print("\n" + "="*80)
    print("VALIDATOR TABLE (All 48 Flights)")
    print("="*80)
    print(header_row)
    print(sep_row)
    for r in results:
        vals = [
            str(r["flight"]).ljust(col_widths[0]),
            str(r["attempt"]).center(col_widths[1]),
            str(r.get("landing_script_initiated", True)).center(col_widths[2]),
            str(r["max_tilt_motion_deg"]).center(col_widths[3]),
            str(r["max_tilt_motion22_deg"]).center(col_widths[4]),
            str(r["max_tilt_inflight_deg"]).center(col_widths[5]),
            str(r["first_tilt_exceed_45_s"]).center(col_widths[6]),
            str(r["first_tilt_z_m"]).center(col_widths[7]),
            str(r["t_return_start_s"]).center(col_widths[8]),
            str(r["landing_command_s"]).center(col_widths[9]),
            str(r["max_tilt_post_landcmd_deg"]).center(col_widths[10]),
            str(r["z_at_max_tilt_post_landcmd"]).center(col_widths[11]),
            str(r["old_max_tilt_z2"]).center(col_widths[12]),
            str(r["first_tilt_any_s"]).center(col_widths[13]),
            str(r["duration_gate"]).ljust(col_widths[14]),
            str(r["envelope_gate"]).ljust(col_widths[15]),
            str(r["failsafe_gate"]).ljust(col_widths[16]),
            str(r["warn_gate"]).ljust(col_widths[17]),
            str(r["headline_eligible"]).center(col_widths[18]),
        ]
        print(" | ".join(vals))

    # Summary
    n_total = len(results)
    n_dur_pass = sum(1 for r in results if r["duration_gate"].startswith("PASS"))
    n_env_pass = sum(1 for r in results if r["envelope_gate"] == "PASS")
    n_fs_pass  = sum(1 for r in results if r["failsafe_gate"].startswith("PASS"))
    n_warn_pass = sum(1 for r in results if r["warn_gate"] == "PASS")
    n_eligible = sum(1 for r in results if r["headline_eligible"])

    print("\nSummary:")
    print(f"  Total flights evaluated: {n_total}")
    print(f"  Duration Gate PASS: {n_dur_pass}/{n_total}")
    print(f"  Envelope Gate PASS: {n_env_pass}/{n_total}")
    print(f"  PX4 Failsafe PASS:  {n_fs_pass}/{n_total}")
    print(f"  Health Warning Gate PASS: {n_warn_pass}/{n_total}")
    print(f"  Headline Eligible: {n_eligible}/{n_total}")

    ineligible = [r for r in results if not r["headline_eligible"]]
    print(f"\nHeadline Ineligible Flights ({len(ineligible)}):")
    for r in ineligible:
        reasons = []
        if not r["duration_gate"].startswith("PASS"):
            reasons.append(f"duration: {r['duration_gate']}")
        if r["envelope_gate"] != "PASS":
            reasons.append(f"envelope: {r['envelope_gate']}")
        if not r["failsafe_gate"].startswith("PASS"):
            reasons.append(f"failsafe: {r['failsafe_gate']}")
        if r["warn_gate"] != "PASS":
            reasons.append(f"health_warn: {r['warn_gate']}")
        try:
            if float(r["max_tilt_motion_deg"]) >= 45.0:
                reasons.append(f"max_tilt_motion: {r['max_tilt_motion_deg']} >= 45.0")
        except Exception:
            pass
        print(f"  - {r['flight']}: {'; '.join(reasons)}")

    false_lsi = [r for r in results if not r.get("landing_script_initiated", True)]
    print(f"\nLanding Script Initiated == False ({len(false_lsi)} flights):")
    for r in false_lsi:
        print(f"  - {r['flight']}: landing_command={r.get('landing_command_s')}, max_tilt_motion={r.get('max_tilt_motion_deg')}")

    # Part 3 Calibration Table
    print("\n" + "="*80)
    print("PART 3 CALIBRATION TABLE (All 48 Flights)")
    print("="*80)
    cal_headers = ["flight", "first_loc_s", "indicator", "t_tilt", "t_vz", "t_yaw", "first_warn", "timing_class"]
    cal_keys = ["flight", "first_loss_of_control_s", "loss_of_control_indicator", "t_onset_tilt", "t_onset_vz", "t_onset_yaw", "first_warn_time", "warning_timing_class"]
    cal_col_widths = [max(len(h), max((len(str(r[k])) for r in results), default=0)) for h, k in zip(cal_headers, cal_keys)]
    print(" | ".join(h.ljust(w) for h, w in zip(cal_headers, cal_col_widths)))
    print("-|-".join("-" * w for w in cal_col_widths))
    for r in results:
        vals = [
            str(r["flight"]).ljust(cal_col_widths[0]),
            str(r["first_loss_of_control_s"]).center(cal_col_widths[1]),
            str(r["loss_of_control_indicator"]).center(cal_col_widths[2]),
            str(r["t_onset_tilt"]).center(cal_col_widths[3]),
            str(r["t_onset_vz"]).center(cal_col_widths[4]),
            str(r["t_onset_yaw"]).center(cal_col_widths[5]),
            str(r["first_warn_time"]).center(cal_col_widths[6]),
            str(r["warning_timing_class"]).ljust(cal_col_widths[7]),
        ]
        print(" | ".join(vals))

    # Calibration Acceptance Check
    flagged_loc = [r["flight"] for r in results if r["first_loss_of_control_s"] != "-"]
    expected_loc_required = {"sweep_G_B_R1", "sweep_G_H_R1", "sweep_G_H_R3", "sweep_M_H_R3"}
    allowed_loc_superset = {"sweep_G_B_R1", "sweep_G_H_R1", "sweep_G_H_R3", "sweep_M_H_R2", "sweep_M_H_R3"}

    print("\nCalibration Acceptance Check:")
    print(f"  Detected loss of control flights: {sorted(flagged_loc)}")
    missing_required = expected_loc_required - set(flagged_loc)
    outside_superset = set(flagged_loc) - allowed_loc_superset
    if missing_required:
        print(f"  [FAIL] Missing required flagged flights: {sorted(missing_required)}")
    elif outside_superset:
        print(f"  [FAIL] Unexpected flights flagged outside allowed superset: {sorted(outside_superset)}")
    else:
        m_h_r3_r = [r for r in results if r["flight"] == "sweep_M_H_R3"][0]
        t_loc_m3 = float(m_h_r3_r["first_loss_of_control_s"])
        print(f"  [PASS] All required flights flagged. M_H_R3 onset at {t_loc_m3:.3f}s (target: 35.75 - 36.0s).")
        if "sweep_M_H_R2" in flagged_loc:
            m_h_r2_r = [r for r in results if r["flight"] == "sweep_M_H_R2"][0]
            print(f"  [INFO] M_H_R2 flagged at {m_h_r2_r['first_loss_of_control_s']}s ({m_h_r2_r['loss_of_control_indicator']}).")
        else:
            print("  [INFO] M_H_R2 did not flag.")

    notes_list = [(r['flight'], r['pre_arm_notes']) for r in results if r['pre_arm_notes']]
    if notes_list:
        print(f"\n  Pre-arm Notes: {notes_list}")


if __name__ == "__main__":
    main()
