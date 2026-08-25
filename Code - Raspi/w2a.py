# W2a Exercise: Digital Output — LED Blink
#
# Wiring (BCM numbering):
#   GPIO18 -> 220ohm resistor -> LED anode(+) -> LED cathode(-) -> GND

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
LED_PIN = 18
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(LED_PIN, GPIO.LOW)
        time.sleep(1)

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    GPIO.cleanup()
    print("GPIO pins have been safely cleaned up!")
