"""
Object Tracker — CSRT + Kalman Filter + Physics.

Usage:
  python tracker.py              # webcam
  python tracker.py video.mp4   # video file

Setup:
  1. Draw a box around the object → ENTER/SPACE
  2. Drag the scale bar to match a known distance, pick unit → ENTER
  3. Click to set the reference position → tracking begins

Controls:
  r = reselect | q = quit
"""

import sys
import time
import cv2
import numpy as np


# ── Kalman Filter ─────────────────────────────────────────────────────────────

def make_kalman(cx, cy):
    """Constant-velocity Kalman filter: state = [x, y, vx, vy]."""
    kf = cv2.KalmanFilter(4, 2)
    dt = 1.0
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]], dtype=np.float32)
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
    return kf


# ── Scale Calibration ─────────────────────────────────────────────────────────

UNIT_OPTIONS = [("1 m", 1.0), ("50 cm", 0.5), ("10 cm", 0.1), ("5 cm", 0.05)]

def calibrate_scale(frame):
    """Let the user drag a bar and pick a unit. Returns metres_per_pixel."""
    win = "Scale Calibration"
    cv2.namedWindow(win)

    bar_len = [200]          # mutable: current bar length in pixels
    dragging = [False]
    unit_idx = [0]           # index into UNIT_OPTIONS

    def on_mouse(event, x, y, flags, _):
        fh = frame.shape[0]
        bar_y = fh - 40
        if event == cv2.EVENT_LBUTTONDOWN and abs(y - bar_y) < 20:
            dragging[0] = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging[0]:
            bar_len[0] = max(30, x - 30)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging[0] = False

    cv2.setMouseCallback(win, on_mouse)
    cv2.createTrackbar("Unit", win, 0, len(UNIT_OPTIONS) - 1, lambda v: None)

    print("Drag the bar to match a known distance. Pick unit. Press ENTER.")

    while True:
        disp = frame.copy()
        unit_idx[0] = cv2.getTrackbarPos("Unit", win)
        label, metres = UNIT_OPTIONS[unit_idx[0]]
        bx0, by = 30, frame.shape[0] - 40
        bx1 = bx0 + bar_len[0]

        # Draw bar
        cv2.line(disp, (bx0, by), (bx1, by), (0, 255, 255), 3)
        cv2.line(disp, (bx0, by - 12), (bx0, by + 12), (0, 255, 255), 2)
        cv2.line(disp, (bx1, by - 12), (bx1, by + 12), (0, 255, 255), 2)
        cv2.putText(disp, f"<-- {label} -->", (bx0, by - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(disp, f"{bar_len[0]} px", (bx0, by + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(disp, "Drag bar | Pick unit | ENTER to confirm",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):  # ENTER or SPACE
            break
        if key == ord('q'):
            return None

    cv2.destroyWindow(win)
    m_per_px = metres / bar_len[0]
    print(f"Scale: {bar_len[0]}px = {label} → {m_per_px:.6f} m/px")
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
    """Tracks position history and computes speed, distance, displacement, acceleration."""

    def __init__(self, ref_px, m_per_px, fps):
        self.ref = np.array(ref_px, dtype=np.float64)
        self.scale = m_per_px
        self.dt = 1.0 / fps
        self.positions = []      # list of (x, y) in pixels
        self.total_dist = 0.0    # cumulative path length in metres

    def update(self, center_px):
        p = np.array(center_px, dtype=np.float64)
        if self.positions:
            self.total_dist += np.linalg.norm(p - self.positions[-1]) * self.scale
        self.positions.append(p)

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

    @property
    def velocity(self):
        n = len(self.positions)
        if n < 2:
            return np.zeros(2), 0.0
        dp = self.positions[-1] - self.positions[-2]
        v_m = self.to_metres(dp) / self.dt
        return v_m, np.linalg.norm(v_m)

    @property
    def acceleration(self):
        n = len(self.positions)
        if n < 3:
            return np.zeros(2), 0.0
        v1 = self.to_metres(self.positions[-1] - self.positions[-2]) / self.dt
        v0 = self.to_metres(self.positions[-2] - self.positions[-3]) / self.dt
        a = (v1 - v0) / self.dt
        return a, np.linalg.norm(a)

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


def draw_hud(frame, status_label, status_color, physics):
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

    # Reference point
    if physics:
        rx, ry = int(physics.ref[0]), int(physics.ref[1])
        cv2.drawMarker(frame, (rx, ry), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open source.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print("Draw a box around the object, then press ENTER/SPACE.")

    while True:
        ret, frame = cap.read()
        if not ret:
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

        # ── Init tracker + Kalman + physics ──
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, roi)
        kf = make_kalman(obj_center[0], obj_center[1])
        physics = Physics(ref, m_per_px, fps)
        physics.update(obj_center)  # initial position
        lost = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Kalman predict
            kf.predict()

            # 2. CSRT update
            ok, bbox = tracker.update(frame)

            if ok:
                x, y, w, h = [int(v) for v in bbox]
                center = (x + w // 2, y + h // 2)

                # Kalman correct
                meas = np.array([[np.float32(center[0])],
                                 [np.float32(center[1])]])
                kf.correct(meas)
                lost = 0

                # Physics uses CSRT (blue box) center
                physics.update(center)

                # Draw CSRT detection (blue)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            else:
                lost += 1

            # Kalman-smoothed box (green/orange)
            kx = int(kf.statePost[0].item())
            ky = int(kf.statePost[1].item())
            half_w, half_h = int(roi[2]) // 2, int(roi[3]) // 2
            color = (0, 255, 0) if lost == 0 else (0, 165, 255)
            cv2.rectangle(frame, (kx - half_w, ky - half_h),
                          (kx + half_w, ky + half_h), color, 2)

            label = "TRACKING" if lost == 0 else ("SEARCHING" if lost < 30 else "LOST")
            draw_hud(frame, label, color, physics)

            cv2.imshow("Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord('r'):
                break  # reselect

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
