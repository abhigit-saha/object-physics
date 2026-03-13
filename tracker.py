"""
Object Tracker — CSRT + Kalman Filter + Physics.

Usage:
  python tracker.py                    # webcam
  python tracker.py video.mp4         # video file (real-time)
  python tracker.py video.mp4 4       # video file, 4× slower

Setup:
  1. Draw a box around the object → ENTER/SPACE
  2. Drag the scale bar to match a known distance, pick unit → ENTER
  3. Click to set the reference position → tracking begins

Controls:
  r = reselect | q = quit
"""

import sys
import time
import os
import subprocess
import tempfile
import cv2
import numpy as np
import csv


# ── Kalman Filter ─────────────────────────────────────────────────────────────

def make_kalman(cx, cy, dt):
    """Constant-velocity Kalman filter: state = [x, y, vx, vy]."""
    # 4- dynamic states (2 pos and 2 vel)
    # 2- measurement dimensions (x, y)
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0], #xnew  = x + vx*dt
        [0, 1, 0, dt], #ynew  = y + vy*dt
        [0, 0, 1,  0], #vxnew = vx
        [0, 0, 0,  1]], dtype=np.float32)
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5.0 #expect the true position and veloicty to deviate from math by 5 unit variance. 
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0 #measurement is at 1 unit variance (mostly accurate csrt measurement)
    kf.errorCovPost = np.eye(4, dtype=np.float32) #inherent uncertainty in the system
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32) #state after making the prediction using both csrt and kalman filter
    return kf


def create_csrt():
    """CSRT with a wider search window."""
    params = cv2.TrackerCSRT_Params()
    params.padding = 15.0
    return cv2.TrackerCSRT_create(params)


# ── Camera Motion Compensator ─────────────────────────────────────────────────

class CameraCompensator:
    """
    Estimates camera motion from background features each frame.

    Maintains a cumulative homography H_world that maps
    world (frame-0) coordinates → current-frame coordinates.

    Use:
      to_world(pt)   – convert current-frame pixel → world pixel
      to_frame(pt)   – convert world pixel → current-frame pixel
    """
    
    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, frame):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.H_world = np.eye(3, dtype=np.float64)  # world → frame-0 (identity)

    def update(self, frame, obj_bbox):
        """
        Estimate camera motion between previous and current frame.
        obj_bbox = (x, y, w, h) of the tracked object — masked out so
        only background (static scene) points are used.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Build mask: everything EXCEPT the object bbox (with some margin)
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        if obj_bbox is not None:
            bx, by, bw, bh = [int(v) for v in obj_bbox]
            margin = 30
            y0 = max(0, by - margin)
            y1 = min(gray.shape[0], by + bh + margin)
            x0 = max(0, bx - margin)
            x1 = min(gray.shape[1], bx + bw + margin)
            mask[y0:y1, x0:x1] = 0

        # Detect background feature points
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=300, qualityLevel=0.01,
            minDistance=7, mask=mask)

        if prev_pts is None or len(prev_pts) < 8:
            # Not enough background features — assume no camera motion
            self.prev_gray = gray
            return

        # Track into current frame
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.LK_PARAMS)
        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        if len(good_prev) < 8:
            self.prev_gray = gray
            return

        # Estimate homography (prev → current) from background points
        H_frame, inlier_mask = cv2.findHomography(
            good_prev, good_next, cv2.RANSAC, ransacReprojThreshold=3.0)

        if H_frame is not None:
            # Accumulate: H_world maps world coords → current frame coords
            self.H_world = H_frame @ self.H_world

        self.prev_gray = gray

    def to_world(self, pt):
        """Transform a current-frame pixel (x, y) → world-frame pixel."""
        H_inv = np.linalg.inv(self.H_world)
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        w = H_inv @ p
        return (w[0] / w[2], w[1] / w[2])

    def to_frame(self, pt):
        """Transform a world pixel (x, y) → current-frame pixel."""
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        w = self.H_world @ p
        return (w[0] / w[2], w[1] / w[2])


# ── Scale Calibration ─────────────────────────────────────────────────────────

def calibrate_scale(frame):
    """Click two points, type the real-world distance in metres. Returns m/px."""
    win = "Scale Calibration"
    cv2.namedWindow(win)
    points = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.setMouseCallback(win, on_mouse)
    print("Click two points on a known distance. Then enter the length in terminal.")

    while len(points) < 2:
        disp = frame.copy()
        cv2.putText(disp, f"Click point {len(points)+1} of 2 on a known distance",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        # Draw placed points
        for i, p in enumerate(points):
            cv2.circle(disp, p, 5, (0, 255, 255), -1)
            cv2.putText(disp, str(i+1), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(win)
            return None

    # Draw final line
    px_dist = np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
    disp = frame.copy()
    cv2.circle(disp, points[0], 5, (0, 255, 255), -1)
    cv2.circle(disp, points[1], 5, (0, 255, 255), -1)
    cv2.line(disp, points[0], points[1], (0, 255, 255), 2)
    mid = ((points[0][0]+points[1][0])//2, (points[0][1]+points[1][1])//2)
    cv2.putText(disp, f"{px_dist:.1f} px", (mid[0]+10, mid[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(disp, "Enter the real length (metres) in terminal",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imshow(win, disp)
    cv2.waitKey(500)  # brief pause so window updates

    # Get length from terminal
    while True:
        try:
            val = float(input(f"Length of that line in metres (pixel dist = {px_dist:.1f}px): "))
            if val <= 0:
                print("Must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a number (e.g. 0.15 for 15cm).")

    cv2.destroyWindow(win)
    m_per_px = val / px_dist
    print(f"Scale: {px_dist:.1f}px = {val}m → {m_per_px:.6f} m/px")
    return m_per_px


# ── Reference Position ────────────────────────────────────────────────────────

def pick_reference(frame, obj_center, m_per_px):
    """Let the user click a reference point. Shows initial displacement."""
    win = "Pick Reference Point"
    ref = [None]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            ref[0] = (x, y)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Click to set the reference position.")

    while ref[0] is None:
        disp = frame.copy()
        # Show object position
        cv2.circle(disp, (int(obj_center[0]), int(obj_center[1])), 6, (255, 0, 0), -1)
        cv2.putText(disp, "Object", (int(obj_center[0]) + 10, int(obj_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        cv2.putText(disp, "Click to set reference position",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return None

    # Show initial displacement
    disp = frame.copy()
    rx, ry = ref[0]
    ox, oy = int(obj_center[0]), int(obj_center[1])
    dx = (ox - rx) * m_per_px
    dy = -(oy - ry) * m_per_px  # flip y: up is positive
    dist = np.hypot(dx, dy)
    cv2.circle(disp, ref[0], 8, (0, 0, 255), -1)
    cv2.circle(disp, (ox, oy), 6, (255, 0, 0), -1)
    cv2.line(disp, ref[0], (ox, oy), (0, 255, 255), 1)
    cv2.putText(disp, f"Initial displacement: {dist:.3f} m", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(disp, "Press any key to start tracking", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow(win, disp)
    cv2.waitKey(0)
    cv2.destroyWindow(win)

    print(f"Reference: ({rx}, {ry})px | Initial displacement: {dist:.3f} m")
    return ref[0]


# ── Physics ───────────────────────────────────────────────────────────────────

class Physics:
    """Tracks position history and computes speed, distance, displacement, acceleration.

    Uses Savitzky-Golay differentiation: fits a least-squares polynomial
    (degree 2) to the last WINDOW position samples, then analytically
    differentiates to get velocity (1st derivative) and acceleration
    (2nd derivative). This is the standard numerical technique used in
    experimental physics — it uses ALL points in the window, not just
    the endpoints, giving much smoother and more accurate derivatives.
    """

    WINDOW = 11  # must be odd, >= 3. Larger = smoother but more lag.

    def __init__(self, ref_px, m_per_px, fps):
        self.ref = np.array(ref_px, dtype=np.float64)
        self.scale = m_per_px
        self.dt = 1.0 / fps
        self.positions = []      # list of (x, y) in pixels
        self.total_dist = 0.0    # cumulative path length in metres
        self.time_elapsed = 0.0  # New: Track time for CSV
        self.history = []        # New: Store frame data

        # Precompute Savitzky-Golay convolution coefficients for 1st and
        # 2nd derivatives, quadratic fit, evaluated at the rightmost point
        # (i.e., causal / one-sided — no future data needed).
        self._sg1 = None  # 1st derivative coefficients
        self._sg2 = None  # 2nd derivative coefficients
        self._build_sg_coeffs(self.WINDOW)

    def _build_sg_coeffs(self, window):
        """
        Build Savitzky-Golay coefficients for 1st and 2nd derivatives.

        For a window of size M and polynomial degree 2, we fit:
            p(t) = a0 + a1*t + a2*t^2
        to the M data points at t = -(M-1), -(M-2), ..., -1, 0
        (where 0 is the current/rightmost point).

        Velocity at t=0:      dp/dt = a1   (in units of px/frame)
        Acceleration at t=0:  d²p/dt² = 2*a2  (in units of px/frame²)

        We solve this via the pseudoinverse of the Vandermonde matrix.
        """
        m = window
        # t values: ..., -2, -1, 0  (current point is at t=0)
        t = np.arange(-(m - 1), 1, dtype=np.float64)  # shape (m,)

        # Vandermonde matrix for degree-2 polynomial
        V = np.column_stack([np.ones(m), t, t**2])  # shape (m, 3)

        # Pseudoinverse: coeffs = (V^T V)^-1 V^T
        C = np.linalg.pinv(V)  # shape (3, m)

        # Row 1 of C gives a1 (1st derivative at t=0), in px/frame
        self._sg1 = C[1, :]  # shape (m,)

        # Row 2 of C gives a2; acceleration = 2*a2, in px/frame²
        self._sg2 = 2.0 * C[2, :]  # shape (m,)

    def update(self, center_px):
        p = np.array(center_px, dtype=np.float64)
        if self.positions:
            self.total_dist += np.linalg.norm(p - self.positions[-1]) * self.scale
        self.positions.append(p)
        self.time_elapsed += self.dt

        # Calculate current state metrics
        d_vec, d_mag = self.displacement
        v_vec, v_mag = self.velocity
        a_vec, a_mag = self.acceleration

        # Log frame data
        self.history.append({
            'Time (s)': round(self.time_elapsed, 4),
            'Pos X (m)': round(d_vec[0], 4),
            'Pos Y (m)': round(d_vec[1], 4),
            'Displacement (m)': round(d_mag, 4),
            'Distance (m)': round(self.total_dist, 4),
            'Vel X (m/s)': round(v_vec[0], 4),
            'Vel Y (m/s)': round(v_vec[1], 4),
            'Speed (m/s)': round(v_mag, 4),
            'Acc X (m/s²)': round(a_vec[0], 4),
            'Acc Y (m/s²)': round(a_vec[1], 4),
            'Accel Mag (m/s²)': round(a_mag, 4)
        })

    def export_csv(self, filename="tracking_data.csv"):
        """Exports the tracked history to a CSV file."""
        if not self.history:
            return
        keys = self.history[0].keys()
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)
        print(f"\n[+] Frame-by-frame data exported to {filename}")

    def to_metres(self, px_vec):
        """Convert pixel vector to metres (y-flipped: up = positive)."""
        return np.array([px_vec[0], -px_vec[1]]) * self.scale

    @property
    def displacement(self):
        if not self.positions:
            return np.zeros(2), 0.0
        d_px = self.positions[-1] - self.ref
        d_m = self.to_metres(d_px)
        return d_m, np.linalg.norm(d_m)

    def _get_window_data(self):
        """Get the last WINDOW positions as (m, 2) array, or None."""
        n = len(self.positions)
        w = min(self.WINDOW, n)
        if w < 3:
            return None, w
        data = np.array(self.positions[-w:])  # shape (w, 2)
        return data, w

    @property
    def velocity(self):
        data, w = self._get_window_data()
        if data is None:
            return np.zeros(2), 0.0

        if w < self.WINDOW:
            # Not enough data yet — use smaller window with fresh coefficients
            t = np.arange(-(w - 1), 1, dtype=np.float64)
            V = np.column_stack([np.ones(w), t, t**2])
            C = np.linalg.pinv(V)
            sg1 = C[1, :]
        else:
            sg1 = self._sg1

        # Velocity in px/frame for each component
        vx_pf = np.dot(sg1, data[:, 0])
        vy_pf = np.dot(sg1, data[:, 1])

        # Convert to m/s
        v_m = self.to_metres(np.array([vx_pf, vy_pf])) / self.dt
        return v_m, np.linalg.norm(v_m)

    @property
    def acceleration(self):
        data, w = self._get_window_data()
        if data is None:
            return np.zeros(2), 0.0

        if w < self.WINDOW:
            t = np.arange(-(w - 1), 1, dtype=np.float64)
            V = np.column_stack([np.ones(w), t, t**2])
            C = np.linalg.pinv(V)
            sg2 = 2.0 * C[2, :]
        else:
            sg2 = self._sg2

        # Acceleration in px/frame² for each component
        ax_pf2 = np.dot(sg2, data[:, 0])
        ay_pf2 = np.dot(sg2, data[:, 1])

        # Convert to m/s²
        a_m = self.to_metres(np.array([ax_pf2, ay_pf2])) / (self.dt ** 2)
        return a_m, np.linalg.norm(a_m)

    def hud_lines(self):
        """Return list of strings for the HUD overlay."""
        lines = []
        d_vec, d_mag = self.displacement
        v_vec, v_mag = self.velocity
        a_vec, a_mag = self.acceleration

        lines.append(f"Displacement: {d_mag:.3f} m  ({d_vec[0]:+.3f}, {d_vec[1]:+.3f})")
        lines.append(f"Distance:     {self.total_dist:.3f} m")
        lines.append(f"Speed:        {v_mag:.3f} m/s  ({v_vec[0]:+.3f}, {v_vec[1]:+.3f})")
        lines.append(f"Accel:        {a_mag:.3f} m/s2 ({a_vec[0]:+.3f}, {a_vec[1]:+.3f})")
        return lines


def draw_hud(frame, status_label, status_color, physics, cam_comp=None):
    """Draw physics HUD overlay on the frame."""
    lines = physics.hud_lines() if physics else []
    # Background
    ph = 40 + len(lines) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (420, ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Status
    cv2.putText(frame, status_label, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    # Physics lines
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, 52 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 150), 1, cv2.LINE_AA)

    # Reference point — transform world-fixed ref into current frame coords
    if physics:
        ref_world = (float(physics.ref[0]), float(physics.ref[1]))
        if cam_comp is not None:
            ref_frame = cam_comp.to_frame(ref_world)
        else:
            ref_frame = ref_world
        rx, ry = int(ref_frame[0]), int(ref_frame[1])
        cv2.drawMarker(frame, (rx, ry), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)


# ── Video Slow-down via ffmpeg ─────────────────────────────────────────────────

def slowdown_video(source, factor):
    """
    Use ffmpeg to create a slow-motion version of the video with
    motion-interpolated frames. Returns (temp_path, original_fps).
    """
    # Get original fps
    cap = cv2.VideoCapture(source)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0
    cap.release()

    # Target fps for the slowed video = orig_fps * factor
    # (more frames per second of real time → smaller motion per frame)
    target_fps = orig_fps * factor

    # PTS multiplier: slow down by factor
    pts_mult = factor

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    print(f"Preprocessing video: {factor}× slow-motion ({orig_fps:.1f} → {target_fps:.0f} fps)...")
    print("This may take a few seconds...")

    cmd = [
        "ffmpeg", "-y", "-i", source,
        "-filter:v", f"minterpolate=fps={int(target_fps)}:mi_mode=blend",
        "-an",  # no audio
        tmp_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        # Fallback: simple frame duplication
        print("Interpolation failed, trying frame duplication...")
        cmd = [
            "ffmpeg", "-y", "-i", source,
            "-filter:v", f"setpts={pts_mult}*PTS",
            "-r", str(int(target_fps)),
            "-an",
            tmp_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    print(f"Slow-motion video ready: {tmp_path}")
    return tmp_path, orig_fps


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    is_video = isinstance(source, str)

    # Slow-down factor from CLI: python tracker.py video.mp4 4
    slowdown = 1.0
    if len(sys.argv) > 2:
        try:
            slowdown = float(sys.argv[2])
            if slowdown < 1:
                slowdown = 1.0
        except ValueError:
            slowdown = 1.0

    tmp_path = None
    orig_fps = None

    if is_video and slowdown > 1:
        # Preprocess: create interpolated slow-mo video
        tmp_path, orig_fps = slowdown_video(source, slowdown)
        cap = cv2.VideoCapture(tmp_path)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Cannot open source.")
        return

    # fps of the file we're actually reading
    playback_fps = cap.get(cv2.CAP_PROP_FPS)
    if playback_fps <= 0:
        playback_fps = 30.0

    # Real-world dt between original video frames
    if orig_fps:
        real_dt = 1.0 / orig_fps
    else:
        real_dt = 1.0 / playback_fps
        orig_fps = playback_fps

    # How many slowed frames correspond to one original frame
    frames_per_orig = slowdown

    if is_video and slowdown > 1:
        print(f"Slow-mo: {slowdown:.0f}× | Original fps: {orig_fps:.1f} | "
              f"Slowed fps: {playback_fps:.0f} | Real dt: {real_dt:.4f}s")

    frame_delay = max(1, int(1000.0 / playback_fps))

    print("Draw a box around the object, then press ENTER/SPACE.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if physics: 
                physics.export_csv()
            return

        # ── Step 1: Select object ──
        roi = cv2.selectROI("Select Object", frame, fromCenter=False)
        cv2.destroyWindow("Select Object")
        if roi[2] == 0 or roi[3] == 0:
            return

        obj_center = (roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)

        # ── Step 2: Scale calibration ──
        m_per_px = calibrate_scale(frame)
        if m_per_px is None:
            return

        # ── Step 3: Reference position ──
        ref = pick_reference(frame, obj_center, m_per_px)
        if ref is None:
            return

        # ── Init tracker + Kalman + physics + camera compensator ──
        tracker = create_csrt()
        tracker.init(frame, roi)
        dt = real_dt  # real-world time between original frames
        cam_comp = CameraCompensator(frame)
        # Object center in world coords (frame 0 = world)
        world_center = obj_center
        kf = make_kalman(world_center[0], world_center[1], dt)
        physics = Physics(ref, m_per_px, orig_fps)
        physics.update(world_center)  # initial position (world frame)
        lost = 0
        last_bbox = roi
        frame_count = 0  # count slowed frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 1. Estimate camera motion (mask out the object)
            cam_comp.update(frame, last_bbox)

            # 2. Kalman predict
            pred = kf.predict()

            # 3. CSRT update (runs every interpolated frame for smooth tracking)
            ok, bbox = tracker.update(frame)

            if ok:
                x, y, w, h = [int(v) for v in bbox]
                frame_center = (x + w // 2, y + h // 2)
                last_bbox = (x, y, w, h)

                # Transform detected center → world coordinates
                world_center = cam_comp.to_world(frame_center)

                # Kalman correct (in world coords)
                meas = np.array([[np.float32(world_center[0])],
                                 [np.float32(world_center[1])]])
                kf.correct(meas)
                lost = 0

                # Physics: update only once per real-world frame interval
                # With slowdown, every N interpolated frames = 1 real frame
                if slowdown <= 1 or frame_count % int(round(frames_per_orig)) == 0:
                    physics.update(world_center)

                # draw csrt detection in current frame coordinates.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            else:
                lost += 1

            # Kalman-smoothed position (in world coords) → project to frame
            kx_w = kf.statePost[0].item()
            ky_w = kf.statePost[1].item()
            kx_f, ky_f = cam_comp.to_frame((kx_w, ky_w))
            kx, ky = int(kx_f), int(ky_f)
            half_w, half_h = int(roi[2]) // 2, int(roi[3]) // 2
            color = (0, 255, 0) if lost == 0 else (0, 165, 255)
            cv2.rectangle(frame, (kx - half_w, ky - half_h),
                          (kx + half_w, ky + half_h), color, 2)

            label = "TRACKING" if lost == 0 else ("SEARCHING" if lost < 30 else "LOST")
            draw_hud(frame, label, color, physics, cam_comp)

            # Show slow-mo info on frame
            if is_video and slowdown > 1:
                cv2.putText(frame, f"Slow-mo: {slowdown:.0f}x", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            cv2.imshow("Tracker", frame)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                if physics: 
                    physics.export_csv()
                cap.release()
                if tmp_path:
                    os.unlink(tmp_path)
                cv2.destroyAllWindows()
                return
            if key == ord('r'):
                if physics: 
                    physics.export_csv()
                break  # reselect

    cap.release()
    if tmp_path:
        os.unlink(tmp_path)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
