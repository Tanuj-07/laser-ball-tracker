"""
calibration.py

Calibration routine for the laser pan-tilt tracker.
Uses brightness threshold + SimpleBlobDetector,
matching laserDetectionTest updated values.
"""

import cv2
import numpy as np
import json
import os
import time

CALIBRATION_FILE = 'calibration_data.json'

# has to match MAX_SPEED_DEG_PER_SEC in the Arduino sketch since there's no
# completion signal from the Arduino — we estimate travel time ourselves
SERVO_MAX_SPEED = 400.0

# extra wait after each move to let the servo fully settle before grabbing a frame
SETTLE_DELAY = 0.2

# shrink the calibration region inward to avoid sketchy corner detections
BOUNDARY_MARGIN_DEG = 3.0

CENTER_PAN  = 90
CENTER_TILT = 135

DEFAULT_DETECTION_PARAMS = {
    'brightness_threshold': 186,
    'blob_min_area':        20,
    'blob_max_area':        48,
    'min_circularity':      0.66,
    'min_inertia':          0.37,
}


def detect_laser(frame, params=None):
    """
    Detect the laser dot in frame using brightness threshold + SimpleBlobDetector.
    Returns (visible, x, y).
    """
    if params is None:
        params = DEFAULT_DETECTION_PARAMS

    # kill saturation since detection is purely brightness-based
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = 0
    v_channel = hsv[:, :, 2]

    _, mask_bright = cv2.threshold(
        v_channel,
        params['brightness_threshold'],
        255,
        cv2.THRESH_BINARY
    )

    # dilate to enlarge the dot so the blob detector can find it reliably
    kernel_small = np.ones((3, 3), np.uint8)
    mask_bright = cv2.dilate(mask_bright, kernel_small, iterations=1)

    # invert since SimpleBlobDetector looks for dark blobs on a light background
    mask_inverted = cv2.bitwise_not(mask_bright)

    # blob detector with area + circularity + inertia filters
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByColor = True;  blob_params.blobColor = 0
    blob_params.filterByArea = True;   blob_params.minArea = params['blob_min_area']
    blob_params.maxArea = params['blob_max_area']
    blob_params.filterByCircularity = True;  blob_params.minCircularity = params['min_circularity']
    blob_params.maxCircularity = 1.0
    blob_params.filterByConvexity = False
    blob_params.filterByInertia = True;  blob_params.minInertiaRatio = params['min_inertia']
    blob_params.maxInertiaRatio = 1.0

    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(mask_inverted)

    if not keypoints:
        return False, None, None

    # if multiple blobs survive, pick the brightest one (usually just one anyway)
    best = min(keypoints, key=lambda kp: v_channel[int(kp.pt[1]), int(kp.pt[0])])

    return True, int(best.pt[0]), int(best.pt[1])


def move_servo(ser, pan, tilt, prev_pan, prev_tilt):
    pan  = int(max(0, min(180, round(pan))))
    tilt = int(max(0, min(180, round(tilt))))

    ser.write(f"{pan},{tilt}\n".encode())

    dist = max(abs(pan - prev_pan), abs(tilt - prev_tilt))
    travel_time = dist / SERVO_MAX_SPEED
    time.sleep(travel_time)

    return pan, tilt


# set to True when Q is pressed in Feed or Mask window
_abort = False


def show_feed(frame, params, visible, x, y, status=""):
    """
    Show Feed and Mask windows. Called every frame during calibration.
    Sets _abort if Q is pressed.
    """
    global _abort

    # rebuild the mask so the Mask window matches what detect_laser actually used
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = 0
    v_channel = hsv[:, :, 2]
    _, mask_bright = cv2.threshold(v_channel, params['brightness_threshold'], 255, cv2.THRESH_BINARY)
    kernel_small = np.ones((3, 3), np.uint8)
    mask_bright = cv2.dilate(mask_bright, kernel_small, iterations=1)
    mask_inverted = cv2.bitwise_not(mask_bright)

    display = frame.copy()
    h, w = frame.shape[:2]

    if visible and x is not None:
        cv2.circle(display, (x, y), 6, (0, 255, 0), 1)
        cv2.circle(display, (x, y), 3, (0, 0, 255), -1)
        cv2.line(display, (x - 20, y), (x + 20, y), (0, 255, 255), 1)
        cv2.line(display, (x, y - 20), (x, y + 20), (0, 255, 255), 1)
        cv2.putText(display, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    else:
        cv2.putText(display, "NO DETECTION", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if status:
        cv2.putText(display, status, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.putText(display, "Q: abort calibration", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    status_color = (0, 200, 0) if visible else (0, 0, 200)
    cv2.rectangle(display, (w - 30, 5), (w - 5, 30), status_color, -1)

    cv2.imshow("Feed", display)
    cv2.imshow("Mask", mask_inverted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        _abort = True


def check_visible_at(cap, ser, pan, tilt, prev_pan, prev_tilt, params=None, status=""):
    global _abort
    if _abort:
        return False, None, None, pan, tilt

    if params is None:
        params = DEFAULT_DETECTION_PARAMS

    pan, tilt = move_servo(ser, pan, tilt, prev_pan, prev_tilt)
    time.sleep(SETTLE_DELAY)

    # try multiple frames — only call it invisible if none of them detect the dot
    SAMPLE_FRAMES = 5
    for _ in range(SAMPLE_FRAMES):
        if _abort:
            return False, None, None, pan, tilt
        time.sleep(0.05)
        ret, frame = cap.read()
        if not ret:
            continue
        visible, x, y = detect_laser(frame, params)
        if visible:
            show_feed(frame, params, visible, x, y, status)
            return True, x, y, pan, tilt

    show_feed(frame, params, False, None, None, status)
    return False, None, None, pan, tilt


def find_boundary(
    cap, ser,
    inside_angle, outside_angle,
    fixed_angle, axis,
    tolerance=1.0, params=None, status=""
):
    prev_pan, prev_tilt = 90, 90

    while abs(outside_angle - inside_angle) > tolerance:
        mid = (inside_angle + outside_angle) / 2.0

        if axis == 'pan':
            visible, x, y, prev_pan, prev_tilt = check_visible_at(
                cap, ser, mid, fixed_angle, prev_pan, prev_tilt, params, status)
        else:
            visible, x, y, prev_pan, prev_tilt = check_visible_at(
                cap, ser, fixed_angle, mid, prev_pan, prev_tilt, params, status)

        if visible:
            inside_angle = mid
        else:
            outside_angle = mid

    return inside_angle


def run_calibration(cap, ser, grid_size=6, tolerance=1.0, params=None):
    if params is None:
        params = DEFAULT_DETECTION_PARAMS

    print("[calibration] Checking laser visibility at center...")

    visible, x, y, pan, tilt = check_visible_at(
        cap, ser, CENTER_PAN, CENTER_TILT, 90, 90, params, "Checking center...")

    if not visible:
        print(f"[calibration] Laser not visible at center ({CENTER_PAN}, {CENTER_TILT}).")
        print("[calibration] Check the Feed and Mask windows to diagnose. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    print("[calibration] Laser visible at center, proceeding.")
    time.sleep(1)

    print("[calibration] Finding left bound...")
    pan_left = find_boundary(cap, ser, CENTER_PAN, 0, CENTER_TILT, 'pan', tolerance, params, "Finding left bound...")
    if _abort: print("[calibration] Aborted."); cv2.destroyAllWindows(); return None

    print("[calibration] Finding right bound...")
    pan_right = find_boundary(cap, ser, CENTER_PAN, 180, CENTER_TILT, 'pan', tolerance, params, "Finding right bound...")
    if _abort: print("[calibration] Aborted."); cv2.destroyAllWindows(); return None

    print("[calibration] Finding top bound...")
    tilt_top = find_boundary(cap, ser, CENTER_TILT, 0, CENTER_PAN, 'tilt', tolerance, params, "Finding top bound...")
    if _abort: print("[calibration] Aborted."); cv2.destroyAllWindows(); return None

    print("[calibration] Finding bottom bound...")
    tilt_bottom = find_boundary(cap, ser, CENTER_TILT, 180, CENTER_PAN, 'tilt', tolerance, params, "Finding bottom bound...")
    if _abort: print("[calibration] Aborted."); cv2.destroyAllWindows(); return None

    print(f"[calibration] Raw bounds: pan [{pan_left:.1f}, {pan_right:.1f}], tilt [{tilt_top:.1f}, {tilt_bottom:.1f}]")

    safe_pan_left = pan_left + BOUNDARY_MARGIN_DEG
    safe_pan_right = pan_right - BOUNDARY_MARGIN_DEG
    safe_tilt_top = tilt_top + BOUNDARY_MARGIN_DEG
    safe_tilt_bottom = tilt_bottom - BOUNDARY_MARGIN_DEG

    if safe_pan_left >= safe_pan_right:
        print("[calibration] Boundary margin too large for pan range.")
        return None

    if safe_tilt_top >= safe_tilt_bottom:
        print("[calibration] Boundary margin too large for tilt range.")
        return None

    print(f"[calibration] Safe bounds: pan [{safe_pan_left:.1f}, {safe_pan_right:.1f}], tilt [{safe_tilt_top:.1f}, {safe_tilt_bottom:.1f}]")

    pan_values = np.linspace(safe_pan_left, safe_pan_right, grid_size)
    tilt_values = np.linspace(safe_tilt_top, safe_tilt_bottom, grid_size)

    grid_points = []
    prev_pan, prev_tilt = pan, tilt

    for row_idx, t in enumerate(tilt_values):
        row = pan_values if row_idx % 2 == 0 else pan_values[::-1]  # snake pattern

        for p in row:
            if _abort:
                print("[calibration] Aborted.")
                cv2.destroyAllWindows()
                return None

            visible, px, py, prev_pan, prev_tilt = check_visible_at(
                cap, ser, p, t, prev_pan, prev_tilt, params,
                f"Grid sweep: pan={p:.1f} tilt={t:.1f}")

            if visible:
                grid_points.append((float(p), float(t), float(px), float(py)))
                print(f"[calibration]   pan={p:.1f} tilt={t:.1f} -> pixel=({px},{py})")
            else:
                print(f"[calibration]   pan={p:.1f} tilt={t:.1f} -> not visible, skipped")

    if len(grid_points) < 6:
        print("[calibration] Not enough valid grid points, calibration failed.")
        return None

    pans  = np.array([g[0] for g in grid_points])
    tilts = np.array([g[1] for g in grid_points])
    xs    = np.array([g[2] for g in grid_points])
    ys    = np.array([g[3] for g in grid_points])

    # bilinear least squares fit: angle = c0 + c1*x + c2*y + c3*x*y
    A = np.column_stack([np.ones_like(xs), xs, ys, xs * ys])
    pan_coeffs,  _, _, _ = np.linalg.lstsq(A, pans,  rcond=None)
    tilt_coeffs, _, _, _ = np.linalg.lstsq(A, tilts, rcond=None)

    result = {
        'bounds': {
            'pan_left':    safe_pan_left,
            'pan_right':   safe_pan_right,
            'tilt_top':    safe_tilt_top,
            'tilt_bottom': safe_tilt_bottom,
        },
        'raw_bounds': {
            'pan_left':    pan_left,
            'pan_right':   pan_right,
            'tilt_top':    tilt_top,
            'tilt_bottom': tilt_bottom,
        },
        'pan_coeffs':       pan_coeffs.tolist(),
        'tilt_coeffs':      tilt_coeffs.tolist(),
        'grid_points':      grid_points,
        'detection_params': params,
    }

    save_calibration_data(result)
    print("[calibration] Done, saved to", CALIBRATION_FILE)
    cv2.destroyAllWindows()
    return result


def pixel_to_angle(cal_data, x, y):
    pan_c  = cal_data['pan_coeffs']
    tilt_c = cal_data['tilt_coeffs']

    pan  = pan_c[0]  + pan_c[1]  * x + pan_c[2]  * y + pan_c[3]  * x * y
    tilt = tilt_c[0] + tilt_c[1] * x + tilt_c[2] * y + tilt_c[3] * x * y

    bounds = cal_data['bounds']
    pan  = max(bounds['pan_left'],  min(bounds['pan_right'],   pan))
    tilt = max(bounds['tilt_top'],  min(bounds['tilt_bottom'], tilt))

    return pan, tilt


def save_calibration_data(data, path=CALIBRATION_FILE):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_calibration_data(path=CALIBRATION_FILE):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    import serial

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    for _ in range(10):  # flush stale buffered frames
        cap.read()

    ser = serial.Serial('COM3', 115200, timeout=0)
    time.sleep(2)  # wait for Arduino to reboot

    # open windows before calibration starts so they're visible from the first move
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        blank = np.zeros((h, w), np.uint8)
        cv2.imshow("Feed", frame)
        cv2.imshow("Mask", blank)
        cv2.waitKey(1)

    print("[calibration] Starting in 2 seconds...")
    time.sleep(2)

    run_calibration(cap, ser, grid_size=10)

    cap.release()
    ser.close()