"""
Projectile Motion Analyser
===========================
Works with both live camera AND video files.

  python main.py --source 0                  # webcam
  python main.py --source video.mp4          # video file

Video mode:
  - Pauses on the first frame.
  - Adjust Scale (px/m) and Mass (g) sliders.
  - Click on the projectile to select it.
  - Video plays and tracks automatically.
  - Saves annotated output video, trajectory CSV, and analysis plots.

Controls:
  click = select object
  r     = re-select
  SPACE = pause / resume
  q     = quit
  s     = screenshot
"""

import argparse
import csv
import sys
import os
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — write to files, don't pop up
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.detection import ObjectIdentifier, ProjectileTracker

# ====================================================================== #
G = 9.81
click_point = None


def on_mouse(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


# ====================================================================== #
#  Physics from a trail of (x_metres, y_metres) points
# ====================================================================== #
def compute_physics(trail_m, dt, mass):
    """Compute physics quantities from last 3 points of metre-scale trail."""
    n = len(trail_m)
    if n < 3:
        return {}

    p0, p1, p2 = trail_m[-3], trail_m[-2], trail_m[-1]
    x, y = p2

    vx = (p2[0] - p0[0]) / (2 * dt)
    vy = (p2[1] - p0[1]) / (2 * dt)
    speed = np.hypot(vx, vy)

    ax = (p2[0] - 2 * p1[0] + p0[0]) / dt**2
    ay = (p2[1] - 2 * p1[1] + p0[1]) / dt**2
    accel = np.hypot(ax, ay)

    total_dist = sum(
        np.hypot(trail_m[i+1][0] - trail_m[i][0],
                 trail_m[i+1][1] - trail_m[i][1])
        for i in range(n - 1)
    )
    straight_dist = np.hypot(x - trail_m[0][0], y - trail_m[0][1])
    height = y - trail_m[0][1]          # positive = up

    ke = 0.5 * mass * speed**2
    pe = mass * G * max(height, 0)
    te = ke + pe
    force = mass * G
    work = -mass * G * height
    momentum = mass * speed

    return dict(
        x=x, y=y, vx=vx, vy=vy, speed=speed,
        ax=ax, ay=ay, accel=accel,
        total_dist=total_dist, straight_dist=straight_dist,
        height=height,
        ke=ke, pe=pe, te=te,
        force=force, work=work, momentum=momentum,
        time=(n - 1) * dt,
    )


# ====================================================================== #
#  Per-frame physics for time-series logging
# ====================================================================== #
def compute_frame_physics(trail_m, idx, dt, mass):
    """Physics at sample index `idx` (needs idx >= 2)."""
    if idx < 2 or idx >= len(trail_m):
        return None
    p0, p1, p2 = trail_m[idx-2], trail_m[idx-1], trail_m[idx]
    vx = (p2[0] - p0[0]) / (2 * dt)
    vy = (p2[1] - p0[1]) / (2 * dt)
    speed = np.hypot(vx, vy)
    ax = (p2[0] - 2*p1[0] + p0[0]) / dt**2
    ay = (p2[1] - 2*p1[1] + p0[1]) / dt**2
    accel = np.hypot(ax, ay)
    height = p2[1] - trail_m[0][1]
    ke = 0.5 * mass * speed**2
    pe = mass * G * max(height, 0)
    return dict(
        t=idx*dt, x=p2[0], y=p2[1],
        vx=vx, vy=vy, speed=speed,
        ax=ax, ay=ay, accel=accel,
        height=height, ke=ke, pe=pe, te=ke+pe,
        momentum=mass*speed,
    )


# ====================================================================== #
#  Drawing helpers
# ====================================================================== #
def draw_trail(frame, trail_px):
    n = len(trail_px)
    for i in range(1, n):
        t = i / n
        color = (int(50*(1-t)), int(255*t), 255)
        cv2.line(frame, trail_px[i-1], trail_px[i], color, 2)


def draw_crosshair(frame, center, radius, found):
    if center is None:
        return
    cx, cy = int(center[0]), int(center[1])
    r = int(radius) if radius else 12
    color = (0, 255, 0) if found else (0, 0, 255)
    cv2.circle(frame, (cx, cy), r, color, 2)
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
    L = r + 10
    cv2.line(frame, (cx-L, cy), (cx-r-3, cy), color, 1)
    cv2.line(frame, (cx+r+3, cy), (cx+L, cy), color, 1)
    cv2.line(frame, (cx, cy-L), (cx, cy-r-3), color, 1)
    cv2.line(frame, (cx, cy+r+3), (cx, cy+L), color, 1)


def draw_hud(frame, data, status, mass, scale_px, trail_px):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    pw = min(380, w - 20)
    ph = min(30 + (len(data) + 4) * 18, h - 20)
    cv2.rectangle(overlay, (5, 5), (5+pw, 5+ph), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y, dy = 22, 18
    green, white, yellow = (100,255,100), (220,220,220), (100,255,255)

    cv2.putText(frame, status, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, yellow, 1, cv2.LINE_AA)
    y += dy
    mpp = 1.0 / scale_px if scale_px > 0 else 0
    cv2.putText(frame, f"Scale: {scale_px} px/m  ({mpp:.5f} m/px)",
                (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, white, 1, cv2.LINE_AA)
    y += dy
    cv2.putText(frame, f"Mass: {mass:.3f} kg",
                (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, white, 1, cv2.LINE_AA)
    y += int(dy * 1.2)
    for key, val in data.items():
        cv2.putText(frame, f"{key}: {val}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, green, 1, cv2.LINE_AA)
        y += dy

    # Scale bar
    bar_y = h - 30
    bar_x0 = 20
    bar_x1 = bar_x0 + scale_px
    cv2.line(frame, (bar_x0, bar_y), (min(bar_x1, w-10), bar_y), (0,255,255), 2)
    cv2.line(frame, (bar_x0, bar_y-8), (bar_x0, bar_y+8), (0,255,255), 2)
    if bar_x1 < w - 5:
        cv2.line(frame, (bar_x1, bar_y-8), (bar_x1, bar_y+8), (0,255,255), 2)
    cv2.putText(frame, "1 metre", (bar_x0+5, bar_y-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)

    draw_trail(frame, trail_px)


# ====================================================================== #
#  Post-analysis plots  (matplotlib → PNG files)
# ====================================================================== #
def generate_analysis_plots(records, save_dir):
    """
    records: list of dicts from compute_frame_physics
    Saves six PNG charts into save_dir.
    """
    if len(records) < 3:
        print("[!] Not enough data for analysis plots.")
        return

    os.makedirs(save_dir, exist_ok=True)

    t   = [r["t"] for r in records]
    x   = [r["x"] for r in records]
    y   = [r["y"] for r in records]
    vx  = [r["vx"] for r in records]
    vy  = [r["vy"] for r in records]
    spd = [r["speed"] for r in records]
    ax  = [r["ax"] for r in records]
    ay  = [r["ay"] for r in records]
    acc = [r["accel"] for r in records]
    ke  = [r["ke"] for r in records]
    pe  = [r["pe"] for r in records]
    te  = [r["te"] for r in records]
    mom = [r["momentum"] for r in records]
    h   = [r["height"] for r in records]

    plt.style.use("dark_background")

    # 1. Trajectory X-Y
    fig, ax1 = plt.subplots(figsize=(8, 6))
    sc = ax1.scatter(x, y, c=t, cmap="plasma", s=15, edgecolors="none")
    ax1.plot(x, y, "w-", alpha=0.3, linewidth=0.8)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Trajectory (colour = time)")
    ax1.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, label="Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "01_trajectory.png"), dpi=150)
    plt.close(fig)

    # 2. Position vs Time
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_x.plot(t, x, "c-", linewidth=1.2)
    ax_x.set_ylabel("X (m)")
    ax_x.set_title("Position vs Time")
    ax_y.plot(t, y, "m-", linewidth=1.2)
    ax_y.set_ylabel("Y (m)")
    ax_y.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "02_position_vs_time.png"), dpi=150)
    plt.close(fig)

    # 3. Velocity vs Time
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, spd, "g-", linewidth=1.2, label="|v|")
    ax1.plot(t, vx, "c--", linewidth=0.8, alpha=0.7, label="vx")
    ax1.plot(t, vy, "m--", linewidth=0.8, alpha=0.7, label="vy")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Velocity vs Time")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "03_velocity_vs_time.png"), dpi=150)
    plt.close(fig)

    # 4. Acceleration vs Time
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, acc, "r-", linewidth=1.2, label="|a|")
    ax1.plot(t, ax, "c--", linewidth=0.8, alpha=0.7, label="ax")
    ax1.plot(t, ay, "m--", linewidth=0.8, alpha=0.7, label="ay")
    ax1.axhline(9.81, color="yellow", linestyle=":", linewidth=0.8, label="g=9.81")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.set_title("Acceleration vs Time")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "04_acceleration_vs_time.png"), dpi=150)
    plt.close(fig)

    # 5. Energy vs Time
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, ke, "r-", linewidth=1.2, label="KE")
    ax1.plot(t, pe, "b-", linewidth=1.2, label="PE")
    ax1.plot(t, te, "w-", linewidth=1.2, label="Total E")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Energy (J)")
    ax1.set_title("Energy vs Time")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "05_energy_vs_time.png"), dpi=150)
    plt.close(fig)

    # 6. Momentum & Height vs Time
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, mom, "g-", linewidth=1.2, label="Momentum")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Momentum (kg·m/s)", color="g")
    ax2 = ax1.twinx()
    ax2.plot(t, h, "y-", linewidth=1.2, label="Height")
    ax2.set_ylabel("Height (m)", color="y")
    ax1.set_title("Momentum & Height vs Time")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "06_momentum_height.png"), dpi=150)
    plt.close(fig)

    print(f"[✓] 6 analysis plots saved to {save_dir}/")


# ====================================================================== #
#  Save trajectory CSV
# ====================================================================== #
def save_csv(records, filepath):
    if not records:
        return
    keys = records[0].keys()
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)
    print(f"[✓] Trajectory CSV saved: {filepath}")


# ====================================================================== #
#  Main
# ====================================================================== #
def main(args):
    global click_point

    # ---------- open source ------------------------------------------- #
    source = int(args.source) if args.source.isdigit() else args.source
    is_video = isinstance(source, str)  # file path → video mode

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[X] Cannot open: {args.source}")
        sys.exit(1)

    if not is_video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0

    skip = max(1, args.skip)
    eff_fps = fps / skip
    dt = 1.0 / eff_fps

    print(f"[i] Source: {'VIDEO ' + str(source) if is_video else 'CAMERA ' + str(source)}")
    print(f"[i] FPS: {fps:.0f}, skip: {skip}, effective: {eff_fps:.1f}, dt: {dt:.4f}s")
    if total_frames > 0:
        print(f"[i] Total frames: {total_frames}  ({total_frames/fps:.1f}s)")

    # ---------- output dir -------------------------------------------- #
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # ---------- window ------------------------------------------------ #
    win = "Projectile Tracker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    cv2.createTrackbar("Scale (px/m)", win, 200, 1000, lambda x: None)
    cv2.createTrackbar("Mass (g)", win, 150, 5000, lambda x: None)

    # ---------- state ------------------------------------------------- #
    identifier = ObjectIdentifier(colour_tolerance=40, min_area=80)
    tracker = ProjectileTracker(dt=dt)
    selected = False
    paused = True                  # start paused (user picks object)
    trail_px = []
    frame_count = 0
    current_frame = None
    video_writer = None
    physics_records = []           # time-series for analysis

    print()
    print("=" * 56)
    print("  1. Adjust SCALE slider (how many pixels = 1 metre)")
    print("  2. Adjust MASS slider (object mass in grams)")
    print("  3. CLICK on the projectile to start tracking")
    print("  SPACE = pause/resume | r = re-select | q = quit")
    print("=" * 56)

    # ---------- main loop --------------------------------------------- #
    while True:
        scale_px = max(1, cv2.getTrackbarPos("Scale (px/m)", win))
        mass_g = max(1, cv2.getTrackbarPos("Mass (g)", win))
        mass_kg = mass_g / 1000.0
        m_per_px = 1.0 / scale_px

        # ---- grab frame ---------------------------------------------- #
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[i] End of video.")
                break
            frame_count += 1
            if frame_count % skip != 0:
                continue
            current_frame = frame.copy()
        else:
            if current_frame is None:
                ret, frame = cap.read()
                if not ret:
                    print("[X] No frames in source.")
                    break
                current_frame = frame.copy()
                frame_count += 1
            frame = current_frame.copy()

        display = frame.copy()

        # ---- handle click -------------------------------------------- #
        if click_point is not None:
            cx, cy = click_point
            click_point = None
            bbox, center, _ = identifier.select_at(frame, cx, cy)
            if bbox is not None:
                tracker = ProjectileTracker(dt=dt)
                tracker.start(frame, bbox, identifier)
                selected = True
                paused = False
                trail_px = []
                physics_records = []

                # Init video writer on first selection
                if is_video and video_writer is None:
                    fh, fw = frame.shape[:2]
                    out_path = os.path.join(out_dir, "tracked_output.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        out_path, fourcc, eff_fps, (fw, fh)
                    )
                    print(f"[i] Writing annotated video → {out_path}")

                print(f"[>>] Tracking started: {center}")
            else:
                print("[!] No object found at click. Try again.")

        # ---- tracking / HUD ----------------------------------------- #
        data = {}
        status = ""
        found = False
        center = None
        radius = None

        if paused and not selected:
            status = "PAUSED — click on the projectile to track"
            fh, fw = display.shape[:2]
            cv2.putText(display, "Click on the projectile",
                        (fw//2 - 140, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2, cv2.LINE_AA)
            if is_video and total_frames > 0:
                progress = f"Frame {frame_count}/{total_frames}"
                cv2.putText(display, progress,
                            (fw//2 - 80, fh//2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200,200,200), 1, cv2.LINE_AA)

        elif selected:
            found, center, radius, bbox, kf_state = tracker.update(frame)

            if found and center is not None:
                trail_px.append((int(center[0]), int(center[1])))
            trail_px = trail_px[-800:]

            if found:
                status = "TRACKING"
            else:
                status = f"SEARCHING ({tracker.lost_count})"
            if tracker.lost_count > 30:
                status = "LOST — press 'r' to re-select"

            # Progress bar for video
            if is_video and total_frames > 0:
                pct = frame_count / total_frames * 100
                status += f"  [{pct:.0f}%]"

            # Convert trail to metres and compute physics
            trail_m = [(p[0]*m_per_px, -p[1]*m_per_px) for p in trail_px]
            physics = compute_physics(trail_m, dt, mass_kg)

            if physics:
                p = physics
                data["Position"]        = f"({p['x']:.3f}, {p['y']:.3f}) m"
                data["Velocity"]        = f"{p['speed']:.3f} m/s"
                data["Vx, Vy"]          = f"({p['vx']:.3f}, {p['vy']:.3f}) m/s"
                data["Acceleration"]    = f"{p['accel']:.3f} m/s²"
                data["Ax, Ay"]          = f"({p['ax']:.3f}, {p['ay']:.3f}) m/s²"
                data["Height"]          = f"{p['height']:.3f} m"
                data["Dist (path)"]     = f"{p['total_dist']:.3f} m"
                data["Dist (straight)"] = f"{p['straight_dist']:.3f} m"
                data["---"]             = "--- Energy ---"
                data["Kinetic Energy"]  = f"{p['ke']:.4f} J"
                data["Potential Energy"]= f"{p['pe']:.4f} J"
                data["Total Energy"]    = f"{p['te']:.4f} J"
                data["--- "]            = "--- Forces ---"
                data["Gravity Force"]   = f"{p['force']:.4f} N"
                data["Work (gravity)"]  = f"{p['work']:.4f} J"
                data["Momentum"]        = f"{p['momentum']:.4f} kg·m/s"
                data["----"]            = "--- Time ---"
                data["Elapsed"]         = f"{p['time']:.2f} s"
                data["Samples"]         = f"{len(trail_px)}"

            # Log per-frame physics for analysis plots
            if found and len(trail_m) >= 3:
                rec = compute_frame_physics(trail_m, len(trail_m)-1, dt, mass_kg)
                if rec:
                    physics_records.append(rec)

            draw_crosshair(display, center, radius, found)

        draw_hud(display, data, status, mass_kg, scale_px, trail_px)
        cv2.imshow(win, display)

        # Write annotated frame to output video
        if video_writer is not None and selected:
            video_writer.write(display)

        # ---- keys ---------------------------------------------------- #
        wait_ms = 1 if not paused else 30
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            paused = True
            selected = False
            tracker = ProjectileTracker(dt=dt)
            identifier = ObjectIdentifier(colour_tolerance=40, min_area=80)
            trail_px = []
            physics_records = []
            print("[i] Re-select mode.")
        elif key == ord(" "):
            paused = not paused
            print(f"[i] {'Paused' if paused else 'Resumed'}")
        elif key == ord("s"):
            fn = os.path.join(out_dir, f"shot_{frame_count:06d}.png")
            cv2.imwrite(fn, display)
            print(f"[SHOT] {fn}")

    # ---------- cleanup ----------------------------------------------- #
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"[✓] Annotated video saved.")
    cv2.destroyAllWindows()

    # ---------- post-analysis ----------------------------------------- #
    if len(physics_records) >= 3:
        # Save CSV
        csv_path = os.path.join(out_dir, "trajectory.csv")
        save_csv(physics_records, csv_path)

        # Generate plots
        plots_dir = os.path.join(out_dir, "plots")
        generate_analysis_plots(physics_records, plots_dir)

        # Print summary
        p = physics_records[-1]
        print()
        print("=" * 50)
        print("  FINAL ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"  Total samples    : {len(physics_records)}")
        print(f"  Duration         : {p['t']:.2f} s")
        print(f"  Final position   : ({p['x']:.3f}, {p['y']:.3f}) m")
        print(f"  Final speed      : {p['speed']:.3f} m/s")
        print(f"  Final accel      : {p['accel']:.3f} m/s²")
        print(f"  Max height       : {max(r['height'] for r in physics_records):.3f} m")
        print(f"  Max speed        : {max(r['speed'] for r in physics_records):.3f} m/s")
        print(f"  Final KE         : {p['ke']:.4f} J")
        print(f"  Final PE         : {p['pe']:.4f} J")
        print(f"  Final Total E    : {p['te']:.4f} J")
        print(f"  Final Momentum   : {p['momentum']:.4f} kg·m/s")
        print("=" * 50)
        print(f"  Output dir: {os.path.abspath(out_dir)}")
        print(f"    - trajectory.csv")
        if is_video:
            print(f"    - tracked_output.mp4")
        print(f"    - plots/  (6 charts)")
        print("=" * 50)
    else:
        print("[!] Not enough tracking data for analysis.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Projectile Tracker — camera or video analysis"
    )
    p.add_argument("--source", default="0",
                   help="Camera index (0,1..) or path to video file")
    p.add_argument("--skip", type=int, default=2,
                   help="Process every N-th frame (default 2)")
    p.add_argument("--output", default="output",
                   help="Output directory for video, CSV, plots")
    args = p.parse_args()
    main(args)