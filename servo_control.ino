#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_MIN 117
#define SERVO_MAX 575
#define PAN_CHANNEL  0
#define TILT_CHANNEL 1

void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setPWMFreq(50);
}

int degreeToPulse(int deg) {
  return SERVO_MIN + (deg / 180.0) * (SERVO_MAX - SERVO_MIN);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    int comma = line.indexOf(',');
    if (comma != -1) {
      int pan  = constrain(line.substring(0, comma).toInt(), 0, 180);
      int tilt = constrain(line.substring(comma + 1).toInt(), 0, 180);
      pwm.setPWM(PAN_CHANNEL,  0, degreeToPulse(pan));
      pwm.setPWM(TILT_CHANNEL, 0, degreeToPulse(tilt));
    }
  }
}