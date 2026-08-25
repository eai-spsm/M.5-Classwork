# W4a Exercise: HC-SR04 Ultrasonic — Print Distance
#
# Wiring (BCM numbering):
#   HC-SR04 VCC  -> 5V
#   HC-SR04 GND  -> GND
#   HC-SR04 Trig -> GPIO23
#   HC-SR04 Echo -> GPIO24   (5V logic on the Echo pin — add a voltage
#                             divider if you're worried about the Pi's 3.3V GPIO)

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
TRIG_PIN = 23
ECHO_PIN = 24

GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

GPIO.output(TRIG_PIN, False)
time.sleep(0.5)

try:
    while True:
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)  # 10 microsecond pulse
        GPIO.output(TRIG_PIN, False)

        pulse_start = time.time()
        pulse_end = time.time()

        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time.time()

        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        if pulse_duration <= 0.00001:
            print("Sensor Error: Pulse timed out. Check hardware/wiring.")
        else:
            speed_of_sound_cm = (331 + 0.6 * 25) * 100
            distance = (pulse_duration * speed_of_sound_cm) // 2
            print(f"Distance: {distance} cm")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nProgram stopped.")
finally:
    GPIO.cleanup()
