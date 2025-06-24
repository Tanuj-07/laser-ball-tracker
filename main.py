import sys
import cv2
import numpy as np
import time
import threading
import serial

from PyQt5.QtWidgets import QApplication
from widgets import ControlPanel
from calibration import load_calibration_data, pixel_to_angle

# ─────────────────────────────────────────────────────────────────────────────
#  BALL TRACKER
#  HSV color detection + Kalman filter + calibration-based servo output
#
#  Requires: calibration_data.json, widgets.py, calibration.py
#  Install:  pip install PyQt5 opencv-python pyserial
#  Run:      python ball_tracker.py
# ─────────────────────────────────────────────────────────────────────────────

# how far ahead (in seconds) to predict ball position to account for pipeline latency
SYSTEM_DELAY = 0.2

# if speed drops below this (px/sec), skip prediction, otherwise the ball drifts
# slowly even when it's sitting still
VELOCITY_THRESHOLD = 8

# morphology kernel for cleaning up the HSV mask
kernel = np.ones((8, 8), np.uint8)


def run_camera(panel, cal_data):
    # try to connect to Arduino, if it fails just run without servo output
    try:
        ser = serial.Serial('COM3', 115200, timeout=0)
        time.sleep(2)  # Arduino resets on serial connect, give it time to boot
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print("[tracker] Serial connected.")
    except Exception as e:
        print(f"Serial error: {e}, running without servo output.")
        ser = None

    cap = cv2.VideoCapture(2)

    cv2.namedWindow("Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    prev_time = time.time()
    dt = 1 / 30  # fallback dt before first frame

    # Kalman filter: 4 state vars (x, y, vx, vy), 2 measurements (x, y)
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.218
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.136
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kalman_initialized = False
    last_radius = 20  # fallback radius before first detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # update dt every frame so the Kalman transition matrix stays accurate
        current_time = time.time()
        dt = max(0.001, min(current_time - prev_time, 0.1))
        prev_time = current_time

        kf.transitionMatrix[0, 2] = dt
        kf.transitionMatrix[1, 3] = dt

        frame = cv2.flip(frame, 1)  # camera is mirrored, flip it back
        h, w = frame.shape[:2]

        # pull latest slider values every frame so changes take effect immediately
        p = panel.get_params()
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * p["p_noise"]
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * p["m_noise"]

        # ── HSV DETECTION ─────────────────────────────────────────────────────
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([p["h_min"], p["s_min"], p["v_min"]])
        upper = np.array([p["h_max"], p["s_max"], p["v_max"]])
        mask  = cv2.inRange(hsv, lower, upper)
        mask  = cv2.erode(mask,  kernel, iterations=1)   # remove small noise blobs
        mask  = cv2.dilate(mask, kernel, iterations=3)   # restore main blob size

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        raw_x, raw_y = 0, 0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 400:  # ignore tiny blobs
                (x, y), radius = cv2.minEnclosingCircle(largest)
                raw_x, raw_y, radius = int(x), int(y), int(radius)
                detected = True
                cv2.circle(frame, (raw_x, raw_y), radius, (255, 0, 0), 1)  # blue = raw detection

        # init Kalman on first valid detection
        if detected and not kalman_initialized:
            kf.statePre  = np.array([[raw_x],[raw_y],[0],[0]], np.float32)
            kf.statePost = np.array([[raw_x],[raw_y],[0],[0]], np.float32)
            kalman_initialized = True

        # if the filter has drifted more than one ball radius from the actual detection,
        # just reset it. handles the case where the ball stops suddenly after a
        # fast flick and the filter keeps coasting in the wrong direction
        if detected and kalman_initialized:
            est_x = float(kf.statePost[0, 0])
            est_y = float(kf.statePost[1, 0])
            drift = ((raw_x - est_x)**2 + (raw_y - est_y)**2) ** 0.5
            if drift > last_radius:
                kf.statePre  = np.array([[raw_x],[raw_y],[0],[0]], np.float32)
                kf.statePost = np.array([[raw_x],[raw_y],[0],[0]], np.float32)

        if detected:
            last_radius = radius

        pred_x = pred_y = 0
        pan = tilt = None

        if kalman_initialized:
            pred = kf.predict()
            state = kf.correct(np.array([[raw_x],[raw_y]], np.float32)) if detected else pred

            x  = float(state[0, 0]); y  = float(state[1, 0])
            vx = float(state[2, 0]); vy = float(state[3, 0])
            speed = (vx**2 + vy**2) ** 0.5

            # only forward-predict if the ball is actually moving
            if speed < VELOCITY_THRESHOLD:
                pred_x, pred_y = int(x), int(y)
            else:
                pred_x = x + vx * SYSTEM_DELAY
                pred_y = y + vy * SYSTEM_DELAY

            # clamp to frame bounds
            pred_x = int(max(0, min(pred_x, w - 1)))
            pred_y = int(max(0, min(pred_y, h - 1)))

            # ── SERVO OUTPUT ──────────────────────────────────────────────────
            if detected and cal_data and ser:
                # mirror x back before conversion, frame is flipped but calibration wasn't
                pan, tilt = pixel_to_angle(cal_data, w - 1 - pred_x, pred_y)
                pan  = max(0, min(180, pan  + p["pan_offset"]))
                tilt = max(0, min(180, tilt - p["tilt_offset"]))  # tilt offset is negated (UI convention)
                msg = f"{int(round(pan))},{int(round(tilt))}\n"
                ser.write(msg.encode())

            # ── DRAW ──────────────────────────────────────────────────────────
            cv2.circle(frame, (pred_x, pred_y), last_radius, (0, 255, 0), 2)   # green = Kalman estimate
            cv2.circle(frame, (pred_x, pred_y), 4, (0, 0, 255), -1)            # red dot = center
            cv2.putText(frame, f"({pred_x}, {pred_y})", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"reset r={last_radius}px", (10, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            if pan is not None:
                cv2.putText(frame, f"pan={pan:.1f} tilt={tilt:.1f}", (10, h - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        fps_display = min(1.0 / dt if dt > 0 else 0, 30)
        panel.set_status(kalman_initialized and detected, pred_x, pred_y, pan, tilt, fps_display)

        cv2.imshow("Feed", frame)
        cv2.imshow("Mask", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == "__main__":
    cal_data = load_calibration_data()
    if cal_data:
        print(f"[tracker] Calibration loaded: pan [{cal_data['bounds']['pan_left']:.1f}, {cal_data['bounds']['pan_right']:.1f}], tilt [{cal_data['bounds']['tilt_top']:.1f}, {cal_data['bounds']['tilt_bottom']:.1f}]")
    else:
        print("[tracker] No calibration file found, servo output disabled.")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    panel = ControlPanel(cal_loaded=cal_data is not None)
    panel.show()

    cam_thread = threading.Thread(target=run_camera, args=(panel, cal_data), daemon=True)
    cam_thread.start()

    sys.exit(app.exec_())