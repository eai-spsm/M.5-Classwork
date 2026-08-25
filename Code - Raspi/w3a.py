# W3a Exercise: Two Buttons, One LED
# Button 1 turns the LED on. While it's held, Button 2 can turn it back off.
#
# Wiring (BCM numbering):
#   GPIO18 -> 220ohm resistor -> LED anode(+) -> LED cathode(-) -> GND
#   GPIO23 -> one leg of Button 1 -> other leg -> GND   (internal pull-up)
#   GPIO24 -> one leg of Button 2 -> other leg -> GND   (internal pull-up)

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
LED_PIN = 18
BTN_PIN1 = 23
BTN_PIN2 = 24

GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BTN_PIN1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BTN_PIN2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.output(LED_PIN, GPIO.LOW)

try:
    while True:
        button_state1 = GPIO.input(BTN_PIN1)
        button_state2 = GPIO.input(BTN_PIN2)

        if button_state1 == GPIO.LOW:
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(0.2)

            if button_state2 == GPIO.LOW:
                GPIO.output(LED_PIN, GPIO.LOW)
                time.sleep(0.2)

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    GPIO.cleanup()
    print("GPIO pins have been safely cleaned up!")
