from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QGroupBox, QFrame, QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# slider definitions for the HSV and Kalman groups
# format: (label, tooltip description, min, max, default, scale)
# scale is what the integer slider value gets divided by to get the real float
SLIDERS = {
    "hsv": [
        ("H Min", "Hue minimum (0-179). Sets the lower color boundary.\nTennis ball yellow-green: ~25-35.", 0, 179, 30,  1),
        ("H Max", "Hue maximum (0-179). Sets the upper color boundary.\nTennis ball yellow-green: ~55-70.", 0, 179, 65,  1),
        ("S Min", "Saturation minimum (0-255). Filters out washed-out/grey areas.\nRaise to reject background noise.", 0, 255, 43,  1),
        ("S Max", "Saturation maximum (0-255). Keep at 255 unless ball has muted color.", 0, 255, 255, 1),
        ("V Min", "Value (brightness) minimum (0-255). Filters out dark regions.\nRaise if shadows are being detected.", 0, 255, 130, 1),
        ("V Max", "Value (brightness) maximum (0-255). Keep at 255 unless ball is overexposed.", 0, 255, 255, 1),
    ],
    "kalman": [
        ("Process Noise",     "How much the Kalman filter trusts its own prediction (0.001-1.0).\nRaise for more responsive tracking. Lower for smoother but laggier output.",  1, 1000, 218, 1000),
        ("Measurement Noise", "How much the filter trusts raw HSV detections (0.001-9.999).\nLower = trusts camera more (less smoothing). Raise to smooth jitter.", 1, 9999, 136, 1000),
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

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default)
        self.slider.setToolTip(description)
        name_label.setToolTip(description)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: #3a3a3a; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #64d2ff; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #2a6fa8; border-radius: 2px;
            }
        """)
        self.slider.valueChanged.connect(self._on_change)

        layout.addLayout(top)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def _fmt(self, val):
        if self.scale == 1:
            return str(val)
        if self.scale == 2:
            return f"{val / self.scale:.1f}"
        return f"{val / self.scale:.3f}"

    def _on_change(self, val):
        self.value_label.setText(self._fmt(val))

    def get_value(self):
        return self.slider.value() / self.scale


class OffsetControl(QWidget):
    # +/- buttons for pan/tilt trim, 0.5 degree steps, range -5 to +5
    def __init__(self, label, step=0.5, min_val=-5.0, max_val=5.0):
        super().__init__()
        self.step = step
        self.min_val = min_val
        self.max_val = max_val
        self.value = 0.0

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        name = QLabel(label)
        name.setFont(QFont("Segoe UI", 10, QFont.Bold))
        name.setStyleSheet("color: #e0e0e0;")
        name.setFixedWidth(70)

        btn_minus = QPushButton("−")
        btn_minus.setFixedSize(32, 32)
        btn_minus.setStyleSheet(self._btn_style())
        btn_minus.clicked.connect(self._decrement)

        self.val_label = QLabel("0.0°")
        self.val_label.setFont(QFont("Segoe UI Mono", 10))
        self.val_label.setStyleSheet("color: #64d2ff;")
        self.val_label.setAlignment(Qt.AlignCenter)
        self.val_label.setFixedWidth(52)

        btn_plus = QPushButton("+")
        btn_plus.setFixedSize(32, 32)
        btn_plus.setStyleSheet(self._btn_style())
        btn_plus.clicked.connect(self._increment)

        btn_reset = QPushButton("↺")
        btn_reset.setFixedSize(32, 32)
        btn_reset.setToolTip("Reset to 0")
        btn_reset.setStyleSheet(self._btn_style("#555555"))
        btn_reset.clicked.connect(self._reset)

        layout.addWidget(name)
        layout.addWidget(btn_minus)
        layout.addWidget(self.val_label)
        layout.addWidget(btn_plus)
        layout.addWidget(btn_reset)
        layout.addStretch()
        self.setLayout(layout)

    def _btn_style(self, color="#2a6fa8"):
        return f"""
            QPushButton {{
                background: {color}; color: #ffffff;
                border: none; border-radius: 6px;
                font: bold 14px "Segoe UI";
            }}
            QPushButton:hover {{ background: #3a7fc8; }}
            QPushButton:pressed {{ background: #1a5f98; }}
        """

    def _increment(self):
        self.value = min(self.max_val, round(self.value + self.step, 1))
        self._update()

    def _decrement(self):
        self.value = max(self.min_val, round(self.value - self.step, 1))
        self._update()

    def _reset(self):
        self.value = 0.0
        self._update()

    def _update(self):
        self.val_label.setText(f"{self.value:+.1f}°")

    def get_value(self):
        return self.value


class ControlPanel(QWidget):
    def __init__(self, cal_loaded):
        super().__init__()
        self.setWindowTitle("Ball Tracker Controls")
        self.setMinimumWidth(440)
        self.setStyleSheet("background-color: #1e1e1e;")

        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Tennis Ball Tracker")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color: #ffffff; padding-bottom: 4px;")
        main_layout.addWidget(title)

        subtitle = QLabel("HSV detection + Kalman filter + calibrated servo output")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #666666; padding-bottom: 8px;")
        main_layout.addWidget(subtitle)

        div = QFrame(); div.setFrameShape(QFrame.HLine); div.setStyleSheet("color: #333333;")
        main_layout.addWidget(div)

        # show whether calibration loaded successfully
        if cal_loaded:
            cal_label = QLabel("✓ Calibration loaded")
            cal_label.setStyleSheet("color: #22c55e; padding: 2px 0;")
        else:
            cal_label = QLabel("⚠ No calibration file, servo output disabled")
            cal_label.setStyleSheet("color: #f59e0b; padding: 2px 0;")
        cal_label.setFont(QFont("Segoe UI", 9))
        main_layout.addWidget(cal_label)

        self.status_label = QLabel("● NO DETECTION")
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #ff4444; padding: 6px 0;")
        main_layout.addWidget(self.status_label)

        self.coord_label = QLabel("")
        self.coord_label.setFont(QFont("Segoe UI Mono", 9))
        self.coord_label.setStyleSheet("color: #aaaaaa;")
        main_layout.addWidget(self.coord_label)

        self.angle_label = QLabel("")
        self.angle_label.setFont(QFont("Segoe UI Mono", 9))
        self.angle_label.setStyleSheet("color: #666666;")
        main_layout.addWidget(self.angle_label)

        self.fps_label = QLabel("")
        self.fps_label.setFont(QFont("Segoe UI Mono", 9))
        self.fps_label.setStyleSheet("color: #555555;")
        main_layout.addWidget(self.fps_label)

        div2 = QFrame(); div2.setFrameShape(QFrame.HLine); div2.setStyleSheet("color: #333333;")
        main_layout.addWidget(div2)

        group_titles = {"hsv": "HSV Color Filter", "kalman": "Kalman Filter"}
        group_colors = {"hsv": "#34d399", "kalman": "#a78bfa"}

        self.sliders = {}
        for group_key, slider_defs in SLIDERS.items():
            group_box = QGroupBox(group_titles[group_key])
            color = group_colors[group_key]
            group_box.setStyleSheet(f"""
                QGroupBox {{
                    font: bold 10pt "Segoe UI"; color: {color};
                    border: 1px solid #333333; border-radius: 6px;
                    margin-top: 10px; padding: 8px;
                }}
                QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
            """)
            group_layout = QVBoxLayout()
            group_layout.setSpacing(4)
            for (label, desc, mn, mx, default, scale) in slider_defs:
                s = LabeledSlider(label, desc, mn, mx, default, scale)
                group_layout.addWidget(s)
                self.sliders[label] = s
            group_box.setLayout(group_layout)
            main_layout.addWidget(group_box)

        offset_box = QGroupBox("Servo Offset")
        offset_box.setStyleSheet("""
            QGroupBox {
                font: bold 10pt "Segoe UI"; color: #fb923c;
                border: 1px solid #333333; border-radius: 6px;
                margin-top: 10px; padding: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        """)
        offset_layout = QVBoxLayout()
        offset_layout.setSpacing(4)
        self.x_offset = OffsetControl("X  (pan)")
        self.y_offset = OffsetControl("Y  (tilt)")
        offset_layout.addWidget(self.x_offset)
        offset_layout.addWidget(self.y_offset)
        offset_box.setLayout(offset_layout)
        main_layout.addWidget(offset_box)

        main_layout.addStretch()
        hint = QLabel("OpenCV windows: Feed · Mask\nPress Q to quit.")
        hint.setFont(QFont("Segoe UI", 8))
        hint.setStyleSheet("color: #555555; padding-top: 8px;")
        main_layout.addWidget(hint)
        self.setLayout(main_layout)

    def get_params(self):
        s = self.sliders
        return {
            "h_min": int(s["H Min"].get_value()), "h_max": int(s["H Max"].get_value()),
            "s_min": int(s["S Min"].get_value()), "s_max": int(s["S Max"].get_value()),
            "v_min": int(s["V Min"].get_value()), "v_max": int(s["V Max"].get_value()),
            "p_noise": s["Process Noise"].get_value(),
            "m_noise": s["Measurement Noise"].get_value(),
            "pan_offset": self.x_offset.get_value(),
            "tilt_offset": self.y_offset.get_value(),
        }

    def set_status(self, detected, pred_x=0, pred_y=0, pan=None, tilt=None, fps=0):
        if detected:
            self.status_label.setText("● DETECTED")
            self.status_label.setStyleSheet("color: #22c55e; padding: 6px 0;")
            self.coord_label.setText(f"pixel  x={pred_x}  y={pred_y}")
            if pan is not None:
                self.angle_label.setText(f"servo  pan={pan:.1f}°  tilt={tilt:.1f}°")
            else:
                self.angle_label.setText("")
        else:
            self.status_label.setText("● NO DETECTION")
            self.status_label.setStyleSheet("color: #ff4444; padding: 6px 0;")
            self.coord_label.setText("")
            self.angle_label.setText("")
        self.fps_label.setText(f"FPS: {fps:.1f}")