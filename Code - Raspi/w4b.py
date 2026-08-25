# W4b Exercise: HC-SR04 Ultrasonic -> LED Range Patterns
# Same sensor as w4a, but drives 5 LEDs into a different pattern
# depending on which distance band the reading falls into.
#
# Wiring (BCM numbering):
#   HC-SR04 Trig -> GPIO23, Echo -> GPIO24 (same as w4a)
#   GPIO5, GPIO6, GPIO13, GPIO19, GPIO26 -> 220ohm resistor each -> LED -> GND

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

led_pins = [5, 6, 13, 19, 26]
TRIG_PIN = 23
ECHO_PIN = 24

for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)

GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

GPIO.output(TRIG_PIN, False)
time.sleep(0.5)


def all_off():
    for pin in led_pins:
        GPIO.output(pin, False)


def all_on():
    for pin in led_pins:
        GPIO.output(pin, True)


# 5 - 20 cm: blink all 5 times
def pattern_blink_all():
    for _ in range(5):
        all_on()
        time.sleep(0.15)
        all_off()
        time.sleep(0.15)


# 21 - 50 cm: run outside-in
def pattern_outside_in():
    all_off()
    for i in range(2):
        GPIO.output(led_pins[i], True)
        GPIO.output(led_pins[-1 - i], True)
        time.sleep(0.2)
    GPIO.output(led_pins[2], True)
    time.sleep(0.3)
    all_off()


# 51 - 100 cm: alternate blinking
def pattern_alternate():
    for _ in range(3):
        for idx, pin in enumerate(led_pins):
            GPIO.output(pin, idx % 2 == 0)
        time.sleep(0.3)
        for idx, pin in enumerate(led_pins):
            GPIO.output(pin, idx % 2 != 0)
        time.sleep(0.3)
    all_off()


# 101 - 150 cm: all on, then off one by one
def pattern_turn_off_sequence():
    all_on()
    time.sleep(0.3)
    for pin in led_pins:
        GPIO.output(pin, False)
        time.sleep(0.15)


print("Distance LED Controller Active. Press Ctrl+C to stop.")

try:
    while True:
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)

        pulse_start = time.time()
        pulse_end = time.time()

        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time.time()

        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start

        if pulse_duration <= 0.00001:
            print("Sensor Error: Pulse timed out.")
            all_off()
        else:
            speed_of_sound_cm = (331 + 0.6 * 25) * 100
            distance = (pulse_duration * speed_of_sound_cm) // 2
            print(f"Distance: {distance} cm")

            if 5 <= distance <= 20:
                print("Action: Blinking 5 times")
                pattern_blink_all()
            elif 21 <= distance <= 50:
                print("Action: Running Outside-In")
                pattern_outside_in()
            elif 51 <= distance <= 100:
                print("Action: Alternate Blinking")
                pattern_alternate()
            elif 101 <= distance <= 150:
                print("Action: Sequential Turn Off")
                pattern_turn_off_sequence()
            else:
                print("Action: All Off")
                all_off()

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nProgram stopped.")
finally:
    GPIO.cleanup()
