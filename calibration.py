import cv2
import numpy as np
import json
import time
import serial

CALIBRATION_FILE = 'calibration_data.json'
SERVO_MAX_SPEED = 400.0

def detect_laser(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None

    candidates = [c for c in contours if 1 <= cv2.contourArea(c) <= 100]
    if not candidates:
        return False, None, None

    smallest = min(candidates, key=cv2.contourArea)
    (x, y), _ = cv2.minEnclosingCircle(smallest)
    return True, int(x), int(y)


def move_servo(ser, pan, tilt, prev_pan, prev_tilt):
    pan  = int(max(0, min(180, round(pan))))
    tilt = int(max(0, min(180, round(tilt))))
    ser.write(f"{pan},{tilt}\n".encode())
    dist = max(abs(pan - prev_pan), abs(tilt - prev_tilt))
    time.sleep(dist / SERVO_MAX_SPEED + 0.3)
    return pan, tilt


def run_calibration(cap, ser, grid_size=6):
    pan_values  = np.linspace(60, 120, grid_size)
    tilt_values = np.linspace(100, 160, grid_size)

    grid_points = []
    prev_pan, prev_tilt = 90, 90

    for row_idx, t in enumerate(tilt_values):
        row = pan_values if row_idx % 2 == 0 else pan_values[::-1]

        for p in row:
            prev_pan, prev_tilt = move_servo(ser, p, t, prev_pan, prev_tilt)

            ret, frame = cap.read()
            if not ret:
                continue

            visible, x, y = detect_laser(frame)

            if visible:
                grid_points.append((float(p), float(t), float(x), float(y)))
                print(f"pan={p:.1f} tilt={t:.1f} -> pixel=({x},{y})")
            else:
                print(f"pan={p:.1f} tilt={t:.1f} -> not detected, skipped")

            cv2.imshow("Feed", frame)
            cv2.waitKey(1)

    if len(grid_points) < 4:
        print("Not enough points, calibration failed.")
        return None

    pans  = np.array([g[0] for g in grid_points])
    tilts = np.array([g[1] for g in grid_points])
    xs    = np.array([g[2] for g in grid_points])
    ys    = np.array([g[3] for g in grid_points])

    A = np.column_stack([np.ones_like(xs), xs, ys, xs * ys])
    pan_coeffs,  _, _, _ = np.linalg.lstsq(A, pans,  rcond=None)
    tilt_coeffs, _, _, _ = np.linalg.lstsq(A, tilts, rcond=None)

    result = {
        'pan_coeffs':  pan_coeffs.tolist(),
        'tilt_coeffs': tilt_coeffs.tolist(),
        'grid_points': grid_points,
    }

    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(result, f, indent=2)

    print("Calibration done, saved to", CALIBRATION_FILE)
    return result


if __name__ == '__main__':
    cap = cv2.VideoCapture(2)
    ser = serial.Serial('COM3', 115200, timeout=0)
    time.sleep(2)

    run_calibration(cap, ser, grid_size=6)

    cap.release()
    ser.close()
    cv2.destroyAllWindows()