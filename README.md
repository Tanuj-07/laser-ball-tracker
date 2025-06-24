# Laser Ball Tracker

Real time tennis ball tracking system using a webcam, OpenCV, and a calibrated pan-tilt laser turret controlled by an Arduino.

A camera detects a tennis ball using HSV color filtering and a Kalman filter predicts its position to compensate for hardware latency. The predicted pixel position is converted to servo angles using a bilinear calibration model (created with calibration routine), making a red laser dot mounted on a pan-tilt turret to follow the ball in real time.

## Demo



## Hardware

**Components:**
- ELEGOO Uno R3
- PCA9685 16-channel PWM servo driver
- 2x SG90 servo motors (pan-tilt kit)
- HiLetGo 650nm 5mW red diode laser
- EMEET C960 1080p webcam
- Dupont wires
- USB-C to USB-A power cable (stripped for servo power, connected to PCA9685)

**Wiring:**
- A5 (Uno R3) -> SCL (PCA9685)
- A4 (Uno R3) -> SDA (PCA9685)
- GND (Uno R3) -> GND (PCA9685)
- GND (Uno R3) -> GND (laser)
- 5V (Uno R3) -> VCC (PCA9685)
- 3.3V (Uno R3) -> laser power
- PCA9685 powered separately via stripped USB-C cable, since two servos exceed what the Uno R3 can supply directly
- Webcam -> computer via USB
- Uno R3 -> computer via USB

## How It Works

**Ball Detection:**
The webcam feed is converted to HSV color space and filtered by hue, saturation, and value ranges tuned for a tennis ball. Morphological erosion and dilation clean up the mask before contour detection finds the ball.

**Kalman Filter:**
A Kalman filter smooths the detected position and predicts where the ball will be after hardware/software latency, compensating for the delay between capture and servo movement.

**Calibration:**
A one-time calibration routine moves the laser across a 10x10 (size is adjustable) grid of servo angles and records the corresponding pixel positions. A bilinear least-squares model is fitted to this data, mapping any and all detected pixel coordinate to the correct pan and tilt angle for the laser to point at the desired location. Must be re-run any time the camera is moved.

**Servo Control:**
The Arduino receives pan and tilt angles over serial and interpolates smoothly to each new position using a fixed max speed, producing semi-fluid motion, independent of camera update speed/framerate.

## Installation

**Install all Python dependencies:**
- `pip install opencv-python pyserial PyQt5 numpy`




**Arduino:**
- Arduino IDE
- Upload `servo_controller.ino` to the Uno R3 using the Arduino IDE. Requires the `Adafruit PWM Servo Driver` library. Install it via the Arduino Library Manager.

## Usage

Run in this order:

1. **Tune laser detection** (ONE TIME ONLY):
- Run python `laserDetectionTest.py` and adjust sliders until the laser dot is detected consistently. 
- Note values and update them in `calibration.py`.

2. **Run Calibration.py:**
- Make sure laser is visible at center.
- No further steps are necessary; file will run and automatically create a json file to store calibration data.

3. **Run the tracker (`main.py`):**
- Adjust pan/tilt offset if needed.

## File Structure

- `main.py` : main ball tracker, runs HSV detection, Kalman filter, and servo output
- `calibration.py` : calibration routine, generates `calibration_data.json`
- `widgets.py` : PyQt5 UI components for the ball tracker control panel
- `laserDetectionTest.py` : one time tuning tool for laser detection parameters, needs to be adjusted based on lighting
- `servo_controller.ino` : Arduino sketch, handles smooth servo interpolation over serial




