#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// servo pulse limits — standard for SG90 / MG90S
#define SERVO_MIN 117  // pulse count for 0 degrees
#define SERVO_MAX 575  // pulse count for 180 degrees

#define PAN_CHANNEL  0  // PCA9685 channel for pan servo
#define TILT_CHANNEL 1  // PCA9685 channel for tilt servo

// === SMOOTHING CONFIG ===
// SG90 is roughly 0.12s per 60 degrees at 4.8V (~400-500 deg/sec) — a bit
// slower and less consistent than the metal-gear MG90S. This caps the rate
// the servo is ASKED to move; set close to its real top speed so motion
// isn't artificially slower than the hardware can go. If it looks sluggish,
// raise this. If it buzzes/jitters at the end of moves, lower it.
const float MAX_SPEED_DEG_PER_SEC = 400.0;

// how often (ms) to recompute position and push a new PWM value. Runs on
// its own clock, independent of how often Python sends a new target — this
// is what makes motion smooth even though the camera only updates ~30x/sec.
// Lower = smoother but more I2C traffic. 5ms (200Hz) is safely within what
// the PCA9685 can handle.
const unsigned long UPDATE_INTERVAL_MS = 5;

// current actual (interpolated) position — what's really being sent to the servo
float currentPan = 90.0;
float currentTilt = 90.0;

// the move currently in progress
float moveStartPan = 90.0, moveStartTilt = 90.0;
float moveTargetPan = 90.0, moveTargetTilt = 90.0;
unsigned long moveStartTime = 0;
unsigned long moveDurationMs = 1;

unsigned long lastUpdateTime = 0;

void setup() {
  Serial.begin(115200);  // open serial connection to laptop
  pwm.begin();
  pwm.setPWMFreq(50);    // servos run at 50Hz
}

// converts a degree (0-180, float) to a PCA9685 pulse count
int degreeToPulse(float deg) {
  return (int)(SERVO_MIN + (deg / 180.0) * (SERVO_MAX - SERVO_MIN));
}

// called whenever a fresh target arrives over serial. Restarts the
// interpolation from wherever the servo ACTUALLY is right now (currentPan/
// currentTilt) rather than the old target — so a new camera frame arriving
// mid-move redirects smoothly instead of causing a jump or a stall.
void startNewMove(float newPan, float newTilt) {
  moveStartPan  = currentPan;
  moveStartTilt = currentTilt;
  moveTargetPan  = newPan;
  moveTargetTilt = newTilt;

  // duration is based on whichever axis has to travel further, at the max
  // speed above. Using ONE shared duration for both axes — rather than
  // capping each axis independently — is what keeps pan and tilt arriving
  // together instead of the shorter move finishing early and producing an
  // L-shaped path. The shorter axis is deliberately slowed down to match.
  float dist = max(abs(moveTargetPan - moveStartPan), abs(moveTargetTilt - moveStartTilt));
  moveDurationMs = (unsigned long)(1000.0 * dist / MAX_SPEED_DEG_PER_SEC);
  if (moveDurationMs < 1) moveDurationMs = 1;
  moveStartTime = millis();
}

void loop() {
  // read any new target from the camera/Python side. This does NOT move the
  // servo directly — it just updates what we're steering toward. Python can
  // send as fast as it wants, no delay needed on that end; this just becomes
  // the new destination for the interpolation below.
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');  // read one line
    int comma = line.indexOf(',');               // find the comma

    if (comma != -1) {
      int pan  = line.substring(0, comma).toInt();   // everything before comma
      int tilt = line.substring(comma + 1).toInt();  // everything after comma

      pan  = constrain(pan,  0, 180);
      tilt = constrain(tilt, 0, 180);

      startNewMove((float)pan, (float)tilt);
    }
  }

  // independent of serial timing: step the interpolation forward and push a
  // new PWM position every UPDATE_INTERVAL_MS. This is what actually
  // produces smooth, diagonal, jitter-free motion — it runs far more often
  // than new camera targets arrive.
  unsigned long now = millis();
  if (now - lastUpdateTime >= UPDATE_INTERVAL_MS) {
    lastUpdateTime = now;

    unsigned long elapsed = now - moveStartTime;
    float frac = (float)elapsed / (float)moveDurationMs;
    if (frac > 1.0) frac = 1.0;  // move complete — hold at target

    currentPan  = moveStartPan  + (moveTargetPan  - moveStartPan)  * frac;
    currentTilt = moveStartTilt + (moveTargetTilt - moveStartTilt) * frac;

    pwm.setPWM(PAN_CHANNEL,  0, degreeToPulse(currentPan));
    pwm.setPWM(TILT_CHANNEL, 0, degreeToPulse(currentTilt));
  }
}