# W2b Exercise: Analog-style Output — PWM Brightness Fade (0-255)
# Pi GPIO has no true analogWrite(), so 0-255 "brightness" is mapped
# to a 0-100% software PWM duty cycle.
#
# Wiring (BCM numbering):
#   GPIO13 -> 220ohm resistor -> LED anode(+) -> LED cathode(-) -> GND
#   (GPIO13 is a hardware PWM-capable pin)

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
LED_PIN = 13
GPIO.setup(LED_PIN, GPIO.OUT)

pwm = GPIO.PWM(LED_PIN, 1000)  # 1000 Hz
pwm.start(0)

try:
    while True:
        # fade up 0 -> 255
        for brightness in range(0, 256, 5):
            duty = (brightness / 255) * 100
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.02)

        # fade down 255 -> 0
        for brightness in range(255, -1, -5):
            duty = (brightness / 255) * 100
            pwm.ChangeDutyCycle(duty)
            time.sleep(0.02)

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    pwm.stop()
    GPIO.cleanup()
    print("GPIO pins have been safely cleaned up!")
