# W3b Exercise: 5-LED Chaser with Step + Auto Buttons
# Button STEP moves one lit LED forward through the row each press.
# Button AUTO runs a self-cycling chase through all 5.
# Both pressed together -> reset all off.
#
# Wiring (BCM numbering):
#   GPIO5, GPIO6, GPIO13, GPIO19, GPIO26 -> 220ohm resistor each -> LED -> GND
#   GPIO23 -> one leg of STEP button -> other leg -> GND   (internal pull-up)
#   GPIO24 -> one leg of AUTO button -> other leg -> GND   (internal pull-up)

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

LED_PINS = [5, 6, 13, 19, 26]
BTN_STEP = 23
BTN_AUTO = 24

for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

GPIO.setup(BTN_STEP, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BTN_AUTO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

current_index = -1


def all_off():
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.LOW)


try:
    while True:
        step_pressed = GPIO.input(BTN_STEP) == GPIO.LOW
        auto_pressed = GPIO.input(BTN_AUTO) == GPIO.LOW

        if step_pressed and auto_pressed:
            print("Both pressed: reset")
            all_off()
            current_index = -1
            time.sleep(0.5)

        elif step_pressed:
            all_off()
            current_index = (current_index + 1) % len(LED_PINS)
            print(f"Step: LED {current_index}")
            GPIO.output(LED_PINS[current_index], GPIO.HIGH)
            time.sleep(0.3)

        elif auto_pressed:
            print("Auto sequence")
            for pin in LED_PINS:
                all_off()
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(0.3)
            all_off()
            current_index = -1

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    GPIO.cleanup()
    print("GPIO pins have been safely cleaned up!")
