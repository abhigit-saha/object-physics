"""
Object Tracker — CSRT + Kalman Filter.

Usage:
  python tracker.py              # webcam
  python tracker.py video.mp4   # video file

Controls:
  ENTER/SPACE = confirm selection
  r           = reselect object
  q           = quit
"""

import sys
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open source.")
        return

    print("Draw a box around the object, then press ENTER/SPACE.")

    while True:
        ret, frame = cap.read()
        if not ret:
            return

        roi = cv2.selectROI("Select Object", frame, fromCenter=False)
        cv2.destroyWindow("Select Object")
        if roi[2] == 0 or roi[3] == 0:
            return

        # ── Init CSRT tracker ──
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, roi)

        # ── Init Kalman ──
        cx, cy = roi[0] + roi[2] / 2, roi[1] + roi[3] / 2
        kf = make_kalman(cx, cy)
        lost = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Kalman predict
            pred = kf.predict()

            # 2. CSRT update (the actual detection)
            ok, bbox = tracker.update(frame)

            if ok:
                x, y, w, h = [int(v) for v in bbox]
                center = (x + w // 2, y + h // 2)

                # 3. Kalman correct with measurement
                meas = np.array([[np.float32(center[0])],
                                 [np.float32(center[1])]])
                kf.correct(meas)
                lost = 0

                # Draw CSRT detection (blue)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            else:
                lost += 1

            # 4. Draw Kalman-smoothed estimate (green/orange)
            kx = int(kf.statePost[0].item())
            ky = int(kf.statePost[1].item())
            half_w, half_h = int(roi[2]) // 2, int(roi[3]) // 2
            color = (0, 255, 0) if lost == 0 else (0, 165, 255)
            cv2.rectangle(frame, (kx - half_w, ky - half_h),
                          (kx + half_w, ky + half_h), color, 2)

            label = "TRACKING" if lost == 0 else ("SEARCHING" if lost < 30 else "LOST")
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
