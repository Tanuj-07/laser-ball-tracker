import sys
import cv2
import numpy as np
import threading

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QGroupBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# laser detection tuning tool — used once to find good detection params,
# then those values get hardcoded into calibration.py
# brightness threshold + SimpleBlobDetector (Method D)
#
# install: pip install PyQt5 opencv-python
# run:     python laser_detection_D_qt.py

# slider definitions: (label, description, min, max, default, scale)
# scale: shown value = slider_int / scale
SLIDERS = {
    "brightness": [
        ("Brightness Threshold", "Pixels brighter than this survive the mask.\nRaise until only the laser dot appears in the Mask window.\nLower if the dot disappears.", 0, 255, 186, 1),
    ],
    "size": [
        ("Min Area (px²)", "Minimum blob size in pixels².\nKeep low so the dot isn't filtered out at distance.", 1, 300, 20, 1),
        ("Max Area (px²)", "Maximum blob size in pixels².\nDrag down to reject large blobs like door cracks or lights.", 1, 2000, 48, 1),
    ],
    "shape": [
        ("Min Circularity", "How round the blob must be (0.0-1.0).\n1.0 = perfect circle. A line of light scores ~0.0.", 0, 100, 66, 100),
        ("Min Convexity", "How convex the blob must be (0.0-1.0).\nCurrently unused, convexity filter is off.", 0, 100, 60, 100),
        ("Min Inertia Ratio", "How elongated the blob can be (0.0-1.0).\n1.0 = circle, 0.0 = line. Helps reject door crack light.", 0, 100, 37, 100),
    ],
}


class LabeledSlider(QWidget):
    def __init__(self, label, description, min_val, max_val, default, scale):
        super().__init__()
        self.scale = scale

        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 6, 0, 6)

        top = QHBoxLayout()
        name_label = QLabel(label)
        name_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        name_label.setStyleSheet("color: #e0e0e0;")

        self.value_label = QLabel(self._fmt(default))
        self.value_label.setFont(QFont("Segoe UI Mono", 10))
        self.value_label.setStyleSheet("color: #64d2ff;")
        self.value_label.setAlignment(Qt.AlignRight)

        top.addWidget(name_label)
        top.addWidget(self.value_label)

        desc_label = QLabel(description)
        desc_label.setFont(QFont("Segoe UI", 8))
        desc_label.setStyleSheet("color: #888888;")
        desc_label.setWordWrap(True)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                background: #3a3a3a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #64d2ff;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #2a6fa8;
                border-radius: 2px;
            }
        """)
        self.slider.valueChanged.connect(self._on_change)

        layout.addLayout(top)
        layout.addWidget(desc_label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def _fmt(self, val):
        if self.scale == 1:
            return str(val)
        return f"{val / self.scale:.2f}"

    def _on_change(self, val):
        self.value_label.setText(self._fmt(val))

    def get_value(self):
        return self.slider.value() / self.scale


class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laser Detection Controls")
        self.setMinimumWidth(420)
        self.setStyleSheet("background-color: #1e1e1e;")

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Laser Dot Detection - Method D")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color: #ffffff; padding-bottom: 4px;")
        main_layout.addWidget(title)

        subtitle = QLabel("Brightness threshold + SimpleBlobDetector")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #666666; padding-bottom: 8px;")
        main_layout.addWidget(subtitle)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #333333;")
        main_layout.addWidget(line)

        self.status_label = QLabel("● NO DETECTION")
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #ff4444; padding: 6px 0;")
        main_layout.addWidget(self.status_label)

        self.coord_label = QLabel("")
        self.coord_label.setFont(QFont("Segoe UI Mono", 9))
        self.coord_label.setStyleSheet("color: #aaaaaa;")
        main_layout.addWidget(self.coord_label)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color: #333333;")
        main_layout.addWidget(line2)

        group_titles = {"brightness": "Brightness Filter", "size": "Blob Size", "shape": "Blob Shape"}
        group_colors = {"brightness": "#a78bfa", "size": "#34d399", "shape": "#64d2ff"}

        self.sliders = {}
        for group_key, slider_defs in SLIDERS.items():
            group_box = QGroupBox(group_titles[group_key])
            color = group_colors[group_key]
            group_box.setStyleSheet(f"""
                QGroupBox {{
                    font: bold 10pt "Segoe UI";
                    color: {color};
                    border: 1px solid #333333;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding: 8px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px;
                }}
            """)
            group_layout = QVBoxLayout()
            group_layout.setSpacing(4)
            for (label, desc, mn, mx, default, scale) in slider_defs:
                s = LabeledSlider(label, desc, mn, mx, default, scale)
                group_layout.addWidget(s)
                self.sliders[label] = s
            group_box.setLayout(group_layout)
            main_layout.addWidget(group_box)

        main_layout.addStretch()
        hint = QLabel("OpenCV windows: Feed (camera) · Mask (threshold)\nPress Q in Feed or Mask to quit.")
        hint.setFont(QFont("Segoe UI", 8))
        hint.setStyleSheet("color: #555555; padding-top: 8px;")
        main_layout.addWidget(hint)
        self.setLayout(main_layout)

    def get_params(self):
        s = self.sliders
        return {
            "v_bright":      int(s["Brightness Threshold"].get_value()),
            "blob_min_area": max(1, int(s["Min Area (px²)"].get_value())),
            "blob_max_area": max(2, int(s["Max Area (px²)"].get_value())),
            "min_circ":      s["Min Circularity"].get_value(),
            "min_conv":      s["Min Convexity"].get_value(),
            "min_inertia":   s["Min Inertia Ratio"].get_value(),
        }

    def set_detection(self, detected, dot_x=0, dot_y=0, dot_r=0):
        if detected:
            self.status_label.setText("● DETECTED")
            self.status_label.setStyleSheet("color: #22c55e; padding: 6px 0;")
            self.coord_label.setText(f"x={dot_x}  y={dot_y}  r={dot_r}")
        else:
            self.status_label.setText("● NO DETECTION")
            self.status_label.setStyleSheet("color: #ff4444; padding: 6px 0;")
            self.coord_label.setText("")


def run_camera(panel):
    cap = cv2.VideoCapture(2)

    # lock exposure so auto-exposure doesn't shift brightness between frames
    # 1 = manual, 3 = auto (varies by driver)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # tune if feed is too dark/bright

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        p = panel.get_params()

        # zero out saturation so detection is purely brightness-based,
        # not affected by color casts or white balance
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = 0
        v_channel = hsv[:, :, 2]
        _, mask_bright = cv2.threshold(v_channel, p["v_bright"], 255, cv2.THRESH_BINARY)

        # dilate to fatten the dot so the blob detector can grab it,
        # and to merge fragmented noise into one large blob that gets rejected by Max Area
        kernel_small = np.ones((3, 3), np.uint8)
        mask_bright = cv2.dilate(mask_bright, kernel_small, iterations=1)

        # invert since SimpleBlobDetector looks for dark blobs on a light background
        mask_inverted = cv2.bitwise_not(mask_bright)

        # circularity and inertia filters reject the door crack light (which is a line,
        # so it scores near 0 on both). laser dot scores much higher on both.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True;  params.blobColor = 0
        params.filterByArea = True;   params.minArea = p["blob_min_area"]; params.maxArea = p["blob_max_area"]
        params.filterByCircularity = True;  params.minCircularity = p["min_circ"]; params.maxCircularity = 1.0
        params.filterByConvexity = False
        params.filterByInertia = True;  params.minInertiaRatio = p["min_inertia"]; params.maxInertiaRatio = 1.0

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask_inverted)

        detected = False
        dot_x = dot_y = dot_r = 0

        if keypoints:
            best = min(keypoints, key=lambda kp: v_channel[int(kp.pt[1]), int(kp.pt[0])])
            dot_x = int(best.pt[0])
            dot_y = int(best.pt[1])
            dot_r = max(int(best.size / 2), 3)
            detected = True

        panel.set_detection(detected, dot_x, dot_y, dot_r)

        display = frame.copy()
        if detected:
            cv2.circle(display, (dot_x, dot_y), dot_r, (0, 255, 0), 1)
            cv2.circle(display, (dot_x, dot_y), 3, (0, 0, 255), -1)
            cv2.line(display, (dot_x - 20, dot_y), (dot_x + 20, dot_y), (0, 255, 255), 1)
            cv2.line(display, (dot_x, dot_y - 20), (dot_x, dot_y + 20), (0, 255, 255), 1)
            cv2.putText(display, f"({dot_x}, {dot_y})  r={dot_r}", (dot_x + 10, dot_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(display, "NO DETECTION", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status_color = (0, 200, 0) if detected else (0, 0, 200)
        cv2.rectangle(display, (w - 30, 5), (w - 5, 30), status_color, -1)

        cv2.imshow("Feed", display)
        cv2.imshow("Mask", mask_inverted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    panel = ControlPanel()
    panel.show()

    # camera loop runs in a background thread so the Qt UI stays responsive
    cam_thread = threading.Thread(target=run_camera, args=(panel,), daemon=True)
    cam_thread.start()

    sys.exit(app.exec_())