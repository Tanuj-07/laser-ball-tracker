import cv2
import numpy as np
import time
import serial

ser = serial.Serial('COM3', 115200, timeout=0)

prev_time = time.time()
dt = 1 / 30

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("H Min", "Trackbars", 30,  179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 65,  179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 43,  255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 130, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("P Noise x1000", "Trackbars", 218,  1000, nothing)
cv2.createTrackbar("M Noise x1000", "Trackbars", 136, 9999, nothing)

SYSTEM_DELAY = 0.2
VELOCITY_THRESHOLD = 8

kernel = np.ones((8, 8), np.uint8)

kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kf.transitionMatrix  = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.218
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.136
kf.errorCovPost      = np.eye(4, dtype=np.float32)
kalman_initialized   = False
last_radius          = 20

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    dt = max(0.001, min(current_time - prev_time, 0.1))
    prev_time = current_time

    kf.transitionMatrix[0, 2] = dt
    kf.transitionMatrix[1, 3] = dt

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    p_noise = max(1, cv2.getTrackbarPos("P Noise x1000", "Trackbars")) / 1000.0
    m_noise = max(1, cv2.getTrackbarPos("M Noise x1000", "Trackbars")) / 1000.0

    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * p_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * m_noise

    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask  = cv2.inRange(hsv, lower, upper)
    mask  = cv2.erode(mask,  kernel, iterations=1)
    mask  = cv2.dilate(mask, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    raw_x, raw_y = 0, 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 400:
            (x, y), radius = cv2.minEnclosingCircle(largest)
            raw_x, raw_y, radius = int(x), int(y), int(radius)
            detected = True
            cv2.circle(frame, (raw_x, raw_y), radius, (255, 0, 0), 1)

    if detected and not kalman_initialized:
        kf.statePre  = np.array([[raw_x],[raw_y],[0],[0]], np.float32)
        kf.statePost = np.array([[raw_x],[raw_y],[0],[0]], np.float32)
        kalman_initialized = True

    if detected:
        last_radius = radius

    if kalman_initialized:
        pred = kf.predict()
        if detected:
            state = kf.correct(np.array([[raw_x],[raw_y]], np.float32))
        else:
            state = pred

        x  = float(state[0, 0]); y  = float(state[1, 0])
        vx = float(state[2, 0]); vy = float(state[3, 0])
        speed = (vx**2 + vy**2) ** 0.5

        if speed < VELOCITY_THRESHOLD:
            pred_x, pred_y = int(x), int(y)
        else:
            pred_x = x + vx * SYSTEM_DELAY
            pred_y = y + vy * SYSTEM_DELAY

        pred_x = int(max(0, min(pred_x, w - 1)))
        pred_y = int(max(0, min(pred_y, h - 1)))

        # basic linear mapping from pixel to servo angle
        pan_angle  = int((pred_x / w) * 180)
        tilt_angle = int((pred_y / h) * 180)
        pan_angle  = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        ser.write(f"{pan_angle},{tilt_angle}\n".encode())

        cv2.circle(frame, (pred_x, pred_y), last_radius, (0, 255, 0), 2)
        cv2.circle(frame, (pred_x, pred_y), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"({pred_x}, {pred_y})", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        fps_display = min(1.0 / dt if dt > 0 else 0, 30)
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Feed", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()