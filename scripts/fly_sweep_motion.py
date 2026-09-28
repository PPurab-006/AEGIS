#!/usr/bin/env python3
"""
Research 2 — Sweep Motion Generator for PX4 SITL in agriculture.world.

Executes a 4x4 grid of flight maneuvers covering yaw-rate and motion combinations:
  Yaw Bins:
    G (gentle):    ~7.5 deg/s
    M (moderate):  ~20.0 deg/s
    A (aggressive):~45.0 deg/s
    E (extreme):   ~90.0 deg/s
  Motion Bins:
    C (cruise):    sustained gentle forward velocity (~0.8 m/s), no braking
    B (braking):   moderate forward velocity with 4 discrete braking events (active reverse pulse + level)
    S (sharp stop):moderate-high velocity with 2 abrupt full stops held 3.0s each
    H (high burst):sustained high velocity (~2.2-2.5 m/s) with 2 bursts to ~3.5 m/s

All attitude commands use SET_ATTITUDE_TARGET (quaternion + closed-loop thrust).
"""

import argparse
import math
import sys
import time
from pymavlink import mavutil
from scipy.spatial.transform import Rotation as R

# Safe envelope in PX4 Local NED coordinates (Spawn origin = 0,0)
LOCAL_X_MIN, LOCAL_X_MAX = -20.0, 20.0
LOCAL_Y_MIN, LOCAL_Y_MAX = -25.0, 25.0
SAFE_Z_MIN, SAFE_Z_MAX = 0.8, 4.5

YAW_RATES = {
    'G': 7.5,
    'M': 20.0,
    'A': 45.0,
    'E': 90.0,
}

BRAKE_WINDOWS_B = [(4.0, 5.5), (8.5, 10.0), (13.0, 14.5), (17.5, 19.0)]
STOP_WINDOWS_S  = [(5.5, 8.5), (13.5, 16.5)]
BURST_WINDOWS_H = [(5.0, 7.5), (13.0, 15.5)]


def main():
    parser = argparse.ArgumentParser(description="Sweep Motion Generator")
    parser.add_argument('--yaw-bin', type=str, required=True, choices=['G', 'M', 'A', 'E'])
    parser.add_argument('--motion-bin', type=str, required=True, choices=['C', 'B', 'S', 'H'])
    parser.add_argument('--duration', type=float, default=22.0, help="Active flight duration in seconds")
    parser.add_argument('--alt-z', type=float, default=2.41, help="Cruise altitude Z (m ENU)")
    args = parser.parse_args()

    yaw_bin = args.yaw_bin
    motion_bin = args.motion_bin
    duration = args.duration
    alt_z = args.alt_z

    yaw_rate_deg_s = YAW_RATES[yaw_bin]

    print(f"[INFO] Sweep Motion: Yaw={yaw_bin} ({yaw_rate_deg_s} deg/s), Motion={motion_bin}, Duration={duration:.1f}s")
    print(f"[INFO] Altitude Target: {alt_z:.2f} m ENU | Local Bounds: X[{LOCAL_X_MIN},{LOCAL_X_MAX}] Y[{LOCAL_Y_MIN},{LOCAL_Y_MAX}]")

    mav_addr = "udpin:0.0.0.0:14540"
    print(f"Connecting to PX4 via MAVLink at {mav_addr}...")
    master = mavutil.mavlink_connection(mav_addr)
    master.wait_heartbeat()
    target_sys = master.target_system
    target_comp = master.target_component
    print(f"Heartbeat received! (SysID: {target_sys}, CompID: {target_comp})")

    def send_setpoint(x_local, y_local, z_enu, yaw=0.0, mask=3576):
        master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            target_sys, target_comp,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mask,
            y_local, x_local, -z_enu,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            yaw, 0.0
        )

    def send_att_setpoint(roll_deg, pitch_deg, yaw_deg, thrust=0.71):
        r = R.from_euler('xyz', [math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)], degrees=False)
        q = r.as_quat()  # x, y, z, w
        q_wxyz = [q[3], q[0], q[1], q[2]]
        # type_mask = 7: ignore body rates, command quaternion + thrust
        master.mav.set_attitude_target_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            target_sys, target_comp,
            7, q_wxyz, 0.0, 0.0, 0.0, float(thrust)
        )

    print("[1/4] Pre-streaming OFFBOARD hover setpoints (Local 0, 0, 2.41m) for 1.5s...")
    t0 = time.time()
    while time.time() - t0 < 1.5:
        send_setpoint(0.0, 0.0, alt_z, yaw=0.0, mask=3576)
        time.sleep(0.05)

    print("[2/4] Arming vehicle...")
    master.mav.command_long_send(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21968, 0, 0, 0, 0, 0
    )
    time.sleep(0.2)

    print("[3/4] Requesting OFFBOARD mode...")
    master.mav.command_long_send(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0, 1, 6, 0, 0, 0, 0, 0
    )
    time.sleep(0.5)

    print("[EVENT: TAKEOFF_START] Climbing to cruise altitude (2.41m ENU)...")
    takeoff_start_time = time.time()
    takeoff_ready = False
    stable_start_time = None
    TAKEOFF_TIMEOUT_SEC = 14.0
    STABILITY_REQUIRED_SEC = 0.5
    TARGET_MIN_ALT_M = 2.0

    while True:
        elapsed_takeoff = time.time() - takeoff_start_time
        if elapsed_takeoff > TAKEOFF_TIMEOUT_SEC:
            print(f"[EVENT: SAFETY_ABORT] Takeoff timeout ({TAKEOFF_TIMEOUT_SEC}s) reached without achieving 2.0m altitude!")
            break

        send_setpoint(0.0, 0.0, alt_z, yaw=0.0, mask=3576)
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
        if msg:
            px_local, py_local, pz_alt = msg.x, msg.y, -msg.z
            if pz_alt >= TARGET_MIN_ALT_M:
                if stable_start_time is None:
                    stable_start_time = time.time()
                    print(f"[EVENT: TAKEOFF_ALTITUDE_UPDATE] Altitude threshold reached ({pz_alt:.2f}m >= {TARGET_MIN_ALT_M}m). Starting 0.5s stability timer...")
                elif (time.time() - stable_start_time) >= STABILITY_REQUIRED_SEC:
                    takeoff_ready = True
                    print(f"[EVENT: TAKEOFF_READY] Cruise altitude stable for {STABILITY_REQUIRED_SEC}s! Total Takeoff Time: {elapsed_takeoff:.2f}s | Pose: Local X={px_local:.2f}m, Local Y={py_local:.2f}m, Alt={pz_alt:.2f}m")
                    break
            else:
                stable_start_time = None

        time.sleep(0.05)

    if not takeoff_ready:
        print("[EVENT: LAND_START] Takeoff readiness failed. Disengaging and landing...")
        master.mav.command_long_send(target_sys, target_comp, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)
        sys.exit(1)

    print(f"\n[EVENT: MOTION_START] Starting motion profile Yaw={yaw_bin} ({yaw_rate_deg_s} deg/s) + Motion={motion_bin} for {duration:.1f}s...\n")
    motion_start_time = time.time()

    curr_x = 0.0
    curr_y = 0.0
    curr_z = alt_z
    curr_vz = 0.0

    while True:
        elapsed = time.time() - motion_start_time
        if elapsed >= duration:
            print(f"[EVENT: MOTION_END] Motion profile completed ({elapsed:.2f}s >= {duration:.1f}s).")
            break

        # Drain all pending LOCAL_POSITION_NED messages to maintain freshest persistent state
        while True:
            msg_pos = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
            if not msg_pos:
                break
            curr_x, curr_y, curr_z = msg_pos.x, msg_pos.y, -msg_pos.z
            curr_vz = -msg_pos.vz

        # 1. Unidirectional continuous yaw angle
        yaw_target_deg = yaw_rate_deg_s * elapsed

        # 2. Pitch angle computation based on motion bin
        roll_cmd_deg = 0.0

        if motion_bin == 'C':
            # Cruise: constant gentle forward pitch
            pitch_cmd_deg = 1.4

        elif motion_bin == 'B':
            # Braking: 4 discrete brake events
            # For each event: active reverse deceleration pulse for 0.4s (-2.0 deg), then level (0.0 deg)
            pitch_cmd_deg = 1.8  # moderate cruise
            for t_start, t_end in BRAKE_WINDOWS_B:
                if t_start <= elapsed <= t_end:
                    dt_in = elapsed - t_start
                    if dt_in < 0.4:
                        pitch_cmd_deg = -2.0  # active deceleration pulse
                    else:
                        pitch_cmd_deg = 0.0   # hold near-zero velocity
                    break

        elif motion_bin == 'S':
            # Sharp Stop: 2 abrupt full stops held 3.0s each
            pitch_cmd_deg = 2.2  # moderate-high forward cruise
            for t_start, t_end in STOP_WINDOWS_S:
                if t_start <= elapsed <= t_end:
                    dt_in = elapsed - t_start
                    if dt_in < 0.5:
                        pitch_cmd_deg = -3.0  # forceful reverse braking pulse
                    else:
                        pitch_cmd_deg = 0.0   # hold full stop
                    break

        elif motion_bin == 'H':
            # High-speed burst: sustained high forward velocity with 2 bursts
            pitch_cmd_deg = 3.5  # fast cruise (~2.2-2.5 m/s)
            for t_start, t_end in BURST_WINDOWS_H:
                if t_start <= elapsed <= t_end:
                    pitch_cmd_deg = 5.5  # high-speed surge (~3.5 m/s)
                    break

        # 3. Altitude feedback (exact formulation from proven F6/F12 flights)
        err_z = alt_z - curr_z
        thrust_cmd = min(0.85, max(0.40, 0.71 + 0.15 * err_z - 0.08 * curr_vz))

        # Send attitude target
        send_att_setpoint(roll_deg=roll_cmd_deg, pitch_deg=pitch_cmd_deg, yaw_deg=yaw_target_deg, thrust=thrust_cmd)

        # 4. Spatial safety bounds checking
        if not (LOCAL_X_MIN <= curr_x <= LOCAL_X_MAX and LOCAL_Y_MIN <= curr_y <= LOCAL_Y_MAX and 0.5 <= curr_z <= 5.0):
            print(f"[EVENT: SAFETY_ABORT] Active spatial bounds breached! Pos: Local X={curr_x:.2f}m, Local Y={curr_y:.2f}m, Alt={curr_z:.2f}m. Disengaging...")
            break

        time.sleep(0.05)

    print("\n[EVENT: RETURN_START] Motion sequence finished. Sending hover setpoint for 1.5s...")
    final_t0 = time.time()
    while time.time() - final_t0 < 1.5:
        send_setpoint(0.0, 0.0, alt_z, yaw=0.0, mask=3576)
        time.sleep(0.05)

    print("[EVENT: LAND_START] Sending LAND command to PX4...")
    master.mav.command_long_send(
        target_sys, target_comp,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    time.sleep(2.0)
    print("[EVENT: LAND_COMPLETE] Land command sent. Exiting motion generator.")


if __name__ == '__main__':
    main()
