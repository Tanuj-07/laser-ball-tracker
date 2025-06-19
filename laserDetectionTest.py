import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  LASER DETECTION TEST FILE
#  Red laser dot on white wall, calibration use.
#
#  Four detection methods. switch with keyboard:
#    1   Method A: HSV color filter only
#    2   Method B: Brightness (V channel) threshold only
#    3   Method C: HSV AND brightness combined
#    4   Method D: Brightness threshold + SimpleBlobDetector  <- DEFAULT (predicted best)
#
#  Press Q to quit.
#
# ─────────────────────────────────────────────────────────────────────────────

def nothing(x):
    pass

# ── WINDOWS ──────────────────────────────────────────────────────────────────
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbars", 800, 400)
cv2.namedWindow("Feed")
cv2.namedWindow("Mask")

# ── HSV TRACKBARS (Methods A, B, C) ──────────────────────────────────────────
# Red wraps around the HSV hue boundary — it lives at BOTH H 0-10 AND H 170-179.
# Two inRange() calls are required; their masks are combined with bitwise_or.
cv2.createTrackbar("H Low Max",    "Trackbars", 10,  30,  nothing)
cv2.createTrackbar("H High Min",   "Trackbars", 160, 179, nothing)
cv2.createTrackbar("S Min",        "Trackbars", 100, 255, nothing)
cv2.createTrackbar("S Max",        "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min HSV",    "Trackbars", 180, 255, nothing)
cv2.createTrackbar("V Max HSV",    "Trackbars", 255, 255, nothing)

# ── BRIGHTNESS TRACKBAR (Methods B, C, D) ────────────────────────────────────
# Laser dot is almost always the brightest point in frame.
# Raise until only the dot survives. Start around 240 for an indoor laser.
cv2.createTrackbar("V Bright Min", "Trackbars", 200, 255, nothing)

# ── BLOB DETECTOR TRACKBARS (Method D) ───────────────────────────────────────
# SimpleBlobDetector works on integer trackbar values — all floats are
# encoded as integer * scale factor, then divided back inside the loop.
#
# Min/Max Area: laser dot at 1-2m is roughly 5-50 px^2. Raise Max Area
#   if the dot is large (close range). Lower Min Area if it disappears.
# Circularity x100: 100 = perfect circle. Laser dot should be ~85-100.
#   Lower if the dot appears slightly elliptical due to angle.
# Convexity x100: 100 = fully convex. Laser dot is always fully convex.
# Inertia x100: measures elongation. 100 = circle, 0 = line.
#   Laser dot should be ~80-100. Lower only if dot is being missed.
cv2.createTrackbar("Blob MinArea",    "Trackbars", 1,   300, nothing)  # px^2 — low: dot is small at 10ft
cv2.createTrackbar("Blob MaxArea",    "Trackbars", 500, 2000, nothing) # px^2 — generous upper bound
cv2.createTrackbar("Circ x100",       "Trackbars", 50,  100, nothing)  # /100 = min circularity — loosened for elliptical dot
cv2.createTrackbar("Conv x100",       "Trackbars", 60,  100, nothing)  # /100 = min convexity — loosened
cv2.createTrackbar("Inertia x100",    "Trackbars", 30,  100, nothing)  # /100 = min inertia ratio — loosened for non-circular dot

# ── STATE ─────────────────────────────────────────────────────────────────────
method = 4  # start on Method D
kernel_small = np.ones((3, 3), np.uint8)  # small kernel — laser dot is tiny

cap = cv2.VideoCapture(2)  # change index if your camera is on a different port

print("=" * 55)
print("  Laser Detection Test")
print("  1 → Method A: HSV only")
print("  2 → Method B: Brightness only")
print("  3 → Method C: HSV + Brightness")
print("  4 → Method D: Blob detector (default, best)")
print("  Q → Quit")
print("=" * 55)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ── READ TRACKBARS ────────────────────────────────────────────────────────
    h_low_max  = cv2.getTrackbarPos("H Low Max",    "Trackbars")
    h_high_min = cv2.getTrackbarPos("H High Min",   "Trackbars")
    s_min      = cv2.getTrackbarPos("S Min",        "Trackbars")
    s_max      = cv2.getTrackbarPos("S Max",        "Trackbars")
    v_min_hsv  = cv2.getTrackbarPos("V Min HSV",    "Trackbars")
    v_max_hsv  = cv2.getTrackbarPos("V Max HSV",    "Trackbars")
    v_bright   = cv2.getTrackbarPos("V Bright Min", "Trackbars")

    blob_min_area  = max(1, cv2.getTrackbarPos("Blob MinArea",  "Trackbars"))
    blob_max_area  = max(2, cv2.getTrackbarPos("Blob MaxArea",  "Trackbars"))
    min_circ       = cv2.getTrackbarPos("Circ x100",  "Trackbars") / 100.0
    min_conv       = cv2.getTrackbarPos("Conv x100",  "Trackbars") / 100.0
    min_inertia    = cv2.getTrackbarPos("Inertia x100","Trackbars") / 100.0

    # ── HSV CONVERSION ────────────────────────────────────────────────────────
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── BUILD COMPONENT MASKS ─────────────────────────────────────────────────

    # HSV mask — two ranges ORed to handle red's wrap-around at hue 0/179
    mask_hsv_low  = cv2.inRange(hsv,
                                np.array([0,          s_min, v_min_hsv]),
                                np.array([h_low_max,  s_max, v_max_hsv]))
    mask_hsv_high = cv2.inRange(hsv,
                                np.array([h_high_min, s_min, v_min_hsv]),
                                np.array([179,        s_max, v_max_hsv]))
    mask_hsv = cv2.bitwise_or(mask_hsv_low, mask_hsv_high)

    # Brightness mask — keep only pixels brighter than threshold
    v_channel = hsv[:, :, 2]
    _, mask_bright = cv2.threshold(v_channel, v_bright, 255, cv2.THRESH_BINARY)

    # Combined mask — must pass both color AND brightness
    mask_combined = cv2.bitwise_and(mask_hsv, mask_bright)

    # ── SELECT ACTIVE MASK AND RUN DETECTION ──────────────────────────────────
    detected  = False
    dot_x = dot_y = dot_r = 0

    if method == 1:
        # ── METHOD A: HSV only ────────────────────────────────────────────────
        mask = mask_hsv.copy()
        mask = cv2.erode(mask,  kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        method_label = "A: HSV only"

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 3:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                dot_x, dot_y, dot_r = int(x), int(y), max(int(radius), 3)
                detected = True

    elif method == 2:
        # ── METHOD B: Brightness only ─────────────────────────────────────────
        mask = mask_bright.copy()
        mask = cv2.erode(mask,  kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        method_label = "B: Brightness only"

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 3:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                dot_x, dot_y, dot_r = int(x), int(y), max(int(radius), 3)
                detected = True

    elif method == 3:
        # ── METHOD C: HSV + Brightness combined ───────────────────────────────
        mask = mask_combined.copy()
        mask = cv2.erode(mask,  kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        method_label = "C: HSV + Brightness"

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 3:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                dot_x, dot_y, dot_r = int(x), int(y), max(int(radius), 3)
                detected = True

    else:
        # ── METHOD D: Brightness threshold + SimpleBlobDetector ───────────────
        # Step 1: threshold first to give the blob detector a clean binary image.
        # Invert it — SimpleBlobDetector looks for DARK blobs on a light background
        # by default when filterByColor is off. Inverting the bright mask means
        # the laser dot becomes a dark blob on a white field, which is what the
        # detector expects. blobColor=0 confirms this.
        mask = mask_bright.copy()
        mask_inverted = cv2.bitwise_not(mask)
        method_label = "D: Blob Detector (best)"

        # Step 2: build detector from current trackbar values every frame
        # so you can tune live without restarting.
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor    = True
        params.blobColor        = 0      # detect dark blobs (laser dot is dark after invert)

        params.filterByArea     = True
        params.minArea          = blob_min_area
        params.maxArea          = blob_max_area

        params.filterByCircularity = True
        params.minCircularity   = min_circ
        params.maxCircularity   = 1.0   # perfect circle is always valid

        params.filterByConvexity = True
        params.minConvexity     = min_conv
        params.maxConvexity     = 1.0

        params.filterByInertia  = True
        params.minInertiaRatio  = min_inertia
        params.maxInertiaRatio  = 1.0

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask_inverted)

        if keypoints:
            # if multiple blobs survive all filters, take the brightest one.
            # "Brightest" = lowest mean value in the original V channel at that location,
            # since we inverted — the original hottest spot maps to the darkest blob.
            # In practice this is rarely more than one keypoint.
            best = min(keypoints,
                       key=lambda kp: v_channel[int(kp.pt[1]), int(kp.pt[0])])
            dot_x  = int(best.pt[0])
            dot_y  = int(best.pt[1])
            dot_r  = max(int(best.size / 2), 3)
            detected = True

        # show the inverted mask in the Mask window for Method D
        mask = mask_inverted

    # ── DRAW ──────────────────────────────────────────────────────────────────
    display = frame.copy()

    if detected:
        # enclosing circle
        cv2.circle(display, (dot_x, dot_y), dot_r, (0, 255, 0), 1)
        # center dot
        cv2.circle(display, (dot_x, dot_y), 3, (0, 0, 255), -1)
        # crosshair — useful for judging accuracy during calibration
        cv2.line(display, (dot_x - 20, dot_y), (dot_x + 20, dot_y), (0, 255, 255), 1)
        cv2.line(display, (dot_x, dot_y - 20), (dot_x, dot_y + 20), (0, 255, 255), 1)

        coord_label = f"({dot_x}, {dot_y})  r={dot_r}"
        cv2.putText(display, coord_label, (dot_x + 10, dot_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    else:
        cv2.putText(display, "NO DETECTION", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # method label — top left
    cv2.putText(display, method_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # status indicator box — top right corner, green = detected, red = not
    status_color = (0, 200, 0) if detected else (0, 0, 200)
    cv2.rectangle(display, (w - 30, 5), (w - 5, 30), status_color, -1)

    cv2.imshow("Feed",    display)
    cv2.imshow("Mask",    mask)

    # ── INPUT ─────────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        method = 1
        print("→ Method A: HSV only")
    elif key == ord('2'):
        method = 2
        print("→ Method B: Brightness only")
    elif key == ord('3'):
        method = 3
        print("→ Method C: HSV + Brightness")
    elif key == ord('4'):
        method = 4
        print("→ Method D: Blob Detector")

cap.release()
cv2.destroyAllWindows()