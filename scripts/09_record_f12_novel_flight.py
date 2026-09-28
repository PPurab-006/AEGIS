#!/usr/bin/env python3
"""
Part 1 of Phase 4 — Record the F12 Novel Maneuver in Gazebo/PX4 SITL.

F12 is a new exploratory family (not in R1's F1-F11 set) designed to
repeatedly trigger the near-zero optical-flow "danger zone" identified in
R2's Phase 2 forensic analysis (Decile 1 velocity: 96.95% failure rate).

Maneuver design:
  - Sustained unidirectional yaw at ~20 deg/s (SET_ATTITUDE_TARGET)
  - Overlaid with 4 discrete braking events (pitch=0, duration=1.5s each)
    at irregular intervals (t=4.5s, 9.0s, 13.5s, 18.5s)
  - Total flight: 22s active window (~600-700 frames at 30 fps)
  - Same agriculture.world, same camera/VO pipeline as all 33 existing flights

Output dataset: results/datasets/p3x_F12_L2_R1/
  - camera_frames.csv, dataset_gt.csv, images/  (raw recording)
  - raw_vo.csv  (from offline run_offline_vo.py RAW mode)
  - eis_gated_vo.csv, gated_dt_def_a_vo.csv  (R1 pipeline compatibility)

STOP GATE: This script produces Part 1 output only. Part 2 (live prediction)
must NOT run until Part 1 is reviewed and approved.

Usage:
    python3 scripts/09_record_f12_novel_flight.py  # from Research2 repo root
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy

# ============================================================
# Configuration
# ============================================================
ROS_REPO   = Path("/home/purab/Purab/Projects/ROS")
PX4_DIR    = os.environ.get("PX4_DIR", str(Path.home() / "PX4-Autopilot"))
MODEL      = "gz_x500_mono_cam"
SPAWN_POSE = "14.0505,-7.5229,0.1076,0,0,0"
WORLD_NAME = "default"

RUN_ID      = "p3x_F12_L2_R1"
DATASET_DIR = ROS_REPO / "results" / "datasets" / RUN_ID
R2_RAW_DIR  = Path(__file__).resolve().parent.parent / "data" / "raw" / RUN_ID
FAMILY      = "F12"
SEVERITY    = 2
DURATION    = 22.0   # seconds active flight

REQUIRED_RAW_VO_COLS = [
    "frame_idx","timestamp_sec","timestamp_nanosec","timestamp_total_sec",
    "pos_x","pos_y","pos_z","rot_x","rot_y","rot_z","rot_w",
    "num_detected","num_matched","num_inliers","num_inliers_E",
    "num_inliers_pose","num_inliers_H","inlier_ratio","feature_survival_rate",
    "feature_vel_mean","feature_vel_max","mean_lk_err","consecutive_low_inliers",
    "failure_triggered_streak","window_low_inlier_pct","failure_triggered_window",
    "rel_tx","rel_ty","rel_tz","rel_rot_deg","eis_dt_ms","eis_crop_pct",
    "eis_warp_deg","eis_cum_warp_deg","eis_gate_scale","eis_yaw_rate_deg",
    "is_r_frame","num_active","num_pending","promotions_this_frame","bd_ratio",
]
REQUIRED_GT_COLS  = [
    "sample_idx","timestamp_sec","timestamp_nanosec","timestamp_total_sec",
    "pos_x","pos_y","pos_z","rot_x","rot_y","rot_z","rot_w",
]
REQUIRED_CAM_COLS = [
    "frame_idx","timestamp_sec","timestamp_nanosec","timestamp_total_sec",
    "filename","width","height",
]


def kill_all():
    print("[CLEANUP] Terminating any existing simulation processes...")
    procs = ["gz-sim-main","gz-sim-gui-client","parameter_bridge",
             "px4","ruby","gz","record_camera_dataset","minimal_vo","fly_phase1_motion"]
    for p in procs:
        subprocess.run(["pkill","-15","-f",p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    for p in procs:
        subprocess.run(["pkill","-9","-f",p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)


def build_env():
    # Load ROS 2 environment if not fully loaded
    env = os.environ.copy()
    if "ROS_DISTRO" not in env:
        try:
            cmd = "source /opt/ros/lyrical/setup.bash && env"
            res = subprocess.check_output(["bash", "-c", cmd]).decode()
            for line in res.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
        except Exception as e:
            print(f"[WARN] Could not source /opt/ros/lyrical/setup.bash: {e}")

    maps_base   = str(ROS_REPO / "configs" / "gazebo_maps")
    col_models  = str(ROS_REPO / "configs" / "gazebo_models_worlds_collection-master" / "models")
    worlds_dir  = str(ROS_REPO / "configs" / "gazebo_models_worlds_collection-master" / "worlds")
    cmn_models  = os.path.join(maps_base, "common_models")
    px4_models  = str(Path(PX4_DIR) / "Tools" / "simulation" / "gz" / "models")

    env.update({
        "__NV_PRIME_RENDER_OFFLOAD":  "1",
        "__GLX_VENDOR_LIBRARY_NAME":  "nvidia",
        "GZ_SIM_RENDER_ENGINE":       "ogre2",
        "PX4_GZ_WORLDS":              f"{worlds_dir}:{maps_base}",
        "PX4_GZ_WORLD":               WORLD_NAME,
        "PX4_GZ_MODEL_POSE":          SPAWN_POSE,
        "GZ_SIM_RESOURCE_PATH":       f"{col_models}:{cmn_models}:{maps_base}:{worlds_dir}:{px4_models}",
        "GAZEBO_MODEL_PATH":          f"{col_models}:{cmn_models}:{maps_base}:{worlds_dir}",
        "PYTHONUNBUFFERED":           "1",
    })
    return env


def wait_for_camera(env, timeout=75):
    print("[WAIT] Waiting for Gazebo camera topic...")
    for sec in range(timeout):
        try:
            out = subprocess.check_output(["gz", "topic", "-l"], env=env, stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                if "camera/image" in line:
                    parts = line.strip().split("/")
                    world = parts[2] if len(parts) > 2 and parts[1] == "world" else WORLD_NAME
                    print(f"   -> Gazebo camera topic detected on world '{world}': {line.strip()}")
                    return world
        except Exception:
            pass
        if sec > 0 and sec % 10 == 0:
            print(f"   ... still waiting ({sec}s / {timeout}s)")
        time.sleep(1.0)
    return None


def verify_basic(ddir):
    gt  = ddir / "dataset_gt.csv"
    cam = ddir / "camera_frames.csv"
    img = ddir / "images"
    if not gt.exists() or not cam.exists():
        return False, "Missing dataset_gt.csv or camera_frames.csv"
    if not img.exists():
        return False, "Missing images/"
    df_gt  = pd.read_csv(gt)
    df_cam = pd.read_csv(cam)
    if len(df_gt) < 100:
        return False, f"Insufficient GT rows: {len(df_gt)}"
    if len(df_cam) < 100:
        return False, f"Insufficient cam rows: {len(df_cam)}"
    t  = df_gt["timestamp_total_sec"].values.astype(float)
    z  = df_gt["pos_z"].values.astype(float)
    ix = np.where(z >= 2.0)[0]
    if len(ix) < 10:
        return False, "Never reached 2.0 m altitude"
    dur = t[ix[-1]] - t[ix[0]]
    if dur < 18.0:
        return False, f"Duration {dur:.1f}s < 18.0s"
    n_img = len([f for f in os.listdir(img) if f.endswith(".png")])
    if n_img < len(df_cam) - 5:
        return False, f"Image count mismatch: {n_img} PNGs vs {len(df_cam)} rows"
    return True, f"PASS (dur={dur:.1f}s, maxZ={z[ix].max():.2f}m, imgs={n_img})"


def verify_schema(ddir):
    rv = ddir / "raw_vo.csv"
    if not rv.exists():
        return False, "raw_vo.csv missing"
    df = pd.read_csv(rv, nrows=2)
    miss = [c for c in REQUIRED_RAW_VO_COLS if c not in df.columns]
    if miss:
        return False, f"Missing columns: {miss}"
    return True, f"OK ({len(df.columns)} columns, matches existing flights)"


def compute_stats(ddir):
    rv = pd.read_csv(ddir / "raw_vo.csv")
    gt = pd.read_csv(ddir / "dataset_gt.csv")
    z  = gt["pos_z"].values
    t  = gt["timestamp_total_sec"].values
    ix = np.where(z >= 2.0)[0]
    t0, t1 = t[ix[0]], t[ix[-1]]
    dur = t1 - t0
    t_vo = rv["timestamp_total_sec"].values
    act  = rv[(t_vo >= t0) & (t_vo <= t1)]
    n    = len(act)
    nf   = (act["num_inliers_pose"] < 8).sum()
    quats = gt[["rot_x","rot_y","rot_z","rot_w"]].values[ix]
    euler = R_scipy.from_quat(quats).as_euler("xyz", degrees=True)
    yaws  = euler[:, 2]
    dyaw  = np.diff(yaws)
    dyaw  = (dyaw + 180) % 360 - 180
    dt    = np.maximum(np.diff(t[ix]), 1e-4)
    rates = np.abs(dyaw / dt)

    # Feature velocity stats during active window
    f_vel_mean = act["feature_vel_mean"].values if "feature_vel_mean" in act.columns else np.array([])

    return {
        "total_frames_active":       int(n),
        "active_duration_s":         float(dur),
        "failure_frames":            int(nf),
        "failure_rate_pct":          float(nf / n * 100 if n > 0 else 0),
        "mean_abs_yaw_rate_deg_s":   float(rates.mean()),
        "median_abs_yaw_rate_deg_s": float(np.median(rates)),
        "max_abs_yaw_rate_deg_s":    float(rates.max()),
        "total_yaw_accumulated_deg": float(np.abs(dyaw).sum()),
        "mean_feature_vel":          float(np.nanmean(f_vel_mean)) if len(f_vel_mean) else 0.0,
        "min_feature_vel":           float(np.nanmin(f_vel_mean)) if len(f_vel_mean) else 0.0,
    }


def run_offline_vo(ddir, env):
    gt  = str(ddir / "dataset_gt.csv")
    vo  = str(ROS_REPO / "src" / "pipelines" / "run_offline_vo.py")
    base = [sys.executable, vo, "--dataset-dir", str(ddir), "--gt-csv", gt]
    print("  [VO 1/3] RAW -> raw_vo.csv")
    subprocess.run(base + ["--output-csv", str(ddir / "raw_vo.csv")], check=True, env=env)
    print("  [VO 2/3] EIS-GATED -> eis_gated_vo.csv")
    subprocess.run(base + ["--output-csv", str(ddir / "eis_gated_vo.csv"),
                           "--eis","--eis-mode","gated","--gate-thresh","15.0"],
                   check=True, env=env)
    print("  [VO 3/3] DELAYED-TRI -> gated_dt_def_a_vo.csv")
    subprocess.run(base + ["--output-csv", str(ddir / "gated_dt_def_a_vo.csv"),
                           "--eis","--eis-mode","gated","--gate-thresh","15.0",
                           "--delayed-triangulation","--r-frame-def","yaw_rate",
                           "--min-non-r-obs","3"],
                   check=True, env=env)
    print(f"  All 3 VO mechanisms done for {RUN_ID}.")


def main():
    print("=================================================================")
    print("Research 2 — Phase 4 Part 1: F12 Novel Maneuver Recording")
    print("=================================================================")
    print(f"  Run ID   : {RUN_ID}")
    print(f"  Maneuver : 20 deg/s sustained yaw (SET_ATTITUDE_TARGET)")
    print(f"             + 4 brake events at t=4.5, 9.0, 13.5, 18.5 s")
    print(f"  Duration : {DURATION}s active window")
    print(f"  Dataset  : {DATASET_DIR}")
    print()

    if DATASET_DIR.exists():
        print(f"[SETUP] Removing existing {DATASET_DIR} for clean re-record...")
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    env = build_env()

    # Symlink agriculture.world as default.sdf in PX4 worlds directory
    agri  = ROS_REPO / "configs" / "gazebo_models_worlds_collection-master" / "worlds" / "agriculture.world"
    dsdf  = Path(PX4_DIR) / "Tools" / "simulation" / "gz" / "worlds" / "default.sdf"
    if not agri.exists():
        print(f"[FATAL] agriculture.world not found at {agri}!"); sys.exit(1)
    subprocess.run(["ln", "-sf", str(agri), str(dsdf)], check=True)
    print(f"[SETUP] Symlinked agriculture.world -> {dsdf}")

    kill_all()

    # 1. PX4 SITL
    print(f"\n[1/5] Launching PX4 SITL (Model: {MODEL}, World: default/agriculture)...")
    px4_log_file = open(DATASET_DIR / "px4_sitl.log", "w")
    px4_proc = subprocess.Popen(
        ["make", "-C", PX4_DIR, "px4_sitl", MODEL],
        env=env, stdout=px4_log_file, stderr=subprocess.STDOUT, cwd=PX4_DIR)

    # 2. Wait for camera
    world = wait_for_camera(env, timeout=75)
    if world is None:
        print("[ERROR] Camera topic timeout. Inspect px4_sitl.log:")
        px4_log_file.flush()
        with open(DATASET_DIR / "px4_sitl.log") as f:
            lines = f.readlines()
            for l in lines[-30:]:
                print("  PX4:", l.rstrip())
        kill_all(); sys.exit(1)

    # 3. ros_gz_bridge
    print(f"\n[3/5] Launching ros_gz_bridge for world '{world}'...")
    bridge_log_file = open(DATASET_DIR / "ros_gz_bridge.log", "w")
    bridge_proc = subprocess.Popen([
        "ros2","run","ros_gz_bridge","parameter_bridge",
        f"/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        f"/world/{world}/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
        f"@sensor_msgs/msg/Image@gz.msgs.Image",
        f"/world/{world}/model/x500_mono_cam_0/link/camera_link/sensor/camera/camera_info"
        f"@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        f"/world/{world}/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
    ], env=env, stdout=bridge_log_file, stderr=subprocess.STDOUT)
    time.sleep(2.5)

    # 4. Dataset recorder
    print(f"\n[4/5] Launching dataset recorder -> {DATASET_DIR}...")
    rec_log_file = open(DATASET_DIR / "recorder.log", "w")
    rec_proc = subprocess.Popen([
        sys.executable,
        str(ROS_REPO / "src" / "archive_phase0" / "prototypes" / "record_camera_dataset.py"),
        "--cam-topic",
        f"/world/{world}/model/x500_mono_cam_0/link/camera_link/sensor/camera/image",
        "--pose-topic",
        f"/world/{world}/dynamic_pose/info",
        "--output-dir", str(DATASET_DIR),
    ], env=env, cwd=str(ROS_REPO), stdout=rec_log_file, stderr=subprocess.STDOUT)
    time.sleep(2.0)

    # 5. Motion
    print(f"\n[5/5] Executing F12 motion ({DURATION}s active duration)...")
    motion_ret = subprocess.run([
        sys.executable,
        str(ROS_REPO / "src" / "flight" / "fly_phase1_motion.py"),
        "--family", FAMILY, "--severity", str(SEVERITY), "--duration", str(DURATION),
    ], env=env, cwd=str(ROS_REPO))

    if motion_ret.returncode != 0:
        print(f"[WARN] Motion generator exited with code {motion_ret.returncode}")

    print("\n[CLEANUP] Flight sequence complete. Waiting 3.0s for data buffer flush...")
    time.sleep(3.0)
    for p, n in [(rec_proc,"recorder"),(bridge_proc,"bridge"),(px4_proc,"px4")]:
        try:
            if p and p.poll() is None:
                p.terminate()
                print(f"  Terminated {n}.")
        except Exception as e:
            print(f"  Warning terminating {n}: {e}")

    px4_log_file.close()
    bridge_log_file.close()
    rec_log_file.close()
    kill_all()

    # Verification
    print("\n=================================================================")
    print("HARD GATE VERIFICATION")
    print("=================================================================")
    ok, msg = verify_basic(DATASET_DIR)
    print(f"  Basic gates : {'PASS' if ok else 'FAIL'} — {msg}")
    if not ok:
        print("FATAL: Hard gate failed. Inspect logs:"); sys.exit(1)

    print("\n=================================================================")
    print("OFFLINE VO EXECUTION")
    print("=================================================================")
    run_offline_vo(DATASET_DIR, env)

    print("\n=================================================================")
    print("SCHEMA VERIFICATION")
    print("=================================================================")
    ok, msg = verify_schema(DATASET_DIR)
    print(f"  raw_vo.csv  : {'PASS' if ok else 'FAIL'} — {msg}")
    if not ok:
        print("FATAL: Schema mismatch."); sys.exit(1)
    dg = pd.read_csv(DATASET_DIR / "dataset_gt.csv",  nrows=2)
    dc = pd.read_csv(DATASET_DIR / "camera_frames.csv", nrows=2)
    gt_miss  = [c for c in REQUIRED_GT_COLS  if c not in dg.columns]
    cam_miss = [c for c in REQUIRED_CAM_COLS if c not in dc.columns]
    print(f"  dataset_gt  : {'PASS' if not gt_miss  else 'FAIL '+str(gt_miss)}")
    print(f"  camera_frames: {'PASS' if not cam_miss else 'FAIL '+str(cam_miss)}")

    # Mirror to Research2 data/raw directory
    print(f"\n[MIRROR] Mirroring dataset files to {R2_RAW_DIR}...")
    R2_RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATASET_DIR / "raw_vo.csv", R2_RAW_DIR / "raw_vo.csv")
    shutil.copy2(DATASET_DIR / "dataset_gt.csv", R2_RAW_DIR / "dataset_gt.csv")
    shutil.copy2(DATASET_DIR / "camera_frames.csv", R2_RAW_DIR / "camera_frames.csv")
    print("  Mirrored raw_vo.csv, dataset_gt.csv, camera_frames.csv.")

    print("\n=================================================================")
    print("BASIC STATS REPORT")
    print("=================================================================")
    s = compute_stats(DATASET_DIR)
    print(f"  Active frames          : {s['total_frames_active']}")
    print(f"  Active window duration : {s['active_duration_s']:.2f} s")
    print(f"  Naive failure frames   : {s['failure_frames']} ({s['failure_rate_pct']:.2f}%)")
    print(f"  Achieved mean |yaw|    : {s['mean_abs_yaw_rate_deg_s']:.2f} deg/s  (commanded: 20.0 deg/s)")
    print(f"  Achieved median |yaw|  : {s['median_abs_yaw_rate_deg_s']:.2f} deg/s")
    print(f"  Max |yaw rate|         : {s['max_abs_yaw_rate_deg_s']:.2f} deg/s")
    print(f"  Total yaw accumulated  : {s['total_yaw_accumulated_deg']:.1f} deg")
    print(f"  Mean feature velocity  : {s['mean_feature_vel']:.2f} px/frame")
    print(f"  Min feature velocity   : {s['min_feature_vel']:.2f} px/frame")

    print("\n=================================================================")
    print("PART 1 COMPLETE — STOP GATE")
    print("=================================================================")
    print(f"Dataset : {DATASET_DIR}")
    for f in sorted(DATASET_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")
    print()
    print(">>> STOP GATE REACHED. Report back to user. Do NOT proceed to Part 2 until approved. <<<")


if __name__ == "__main__":
    main()
