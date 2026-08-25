# W5a Exercise: DC Motor Driver — Button Forward/Backward
#
# Wiring (BCM numbering):
#   Driver ENA -> GPIO13 (PWM speed control)
#   Driver IN1 -> GPIO18
#   Driver IN2 -> GPIO12
#   Driver GND -> Pi GND (shared ground with motor supply)
#   Motor power -> separate motor supply, NOT the Pi's 5V pin
#   GPIO19 -> one leg of FORWARD button -> other leg -> GND   (internal pull-up)
#   GPIO26 -> one leg of BACKWARD button -> other leg -> GND  (internal pull-up)

import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1 = 18
IN2 = 12
ENA = 13
BTN_PIN1 = 19
BTN_PIN2 = 26

GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(BTN_PIN1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BTN_PIN2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm_motor = GPIO.PWM(ENA, 1000)
pwm_motor.start(0)


def motor_forward(speed=40):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm_motor.ChangeDutyCycle(speed)


def motor_backward(speed=40):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm_motor.ChangeDutyCycle(speed)


def motor_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwm_motor.ChangeDutyCycle(0)


try:
    print("Hold FORWARD or BACKWARD button to drive. Ctrl+C to quit.")

    while True:
        button_state1 = GPIO.input(BTN_PIN1)
        button_state2 = GPIO.input(BTN_PIN2)

        if button_state1 == GPIO.LOW:
            print("F")
            motor_forward(40)
        elif button_state2 == GPIO.LOW:
            print("B")
            motor_backward(40)
        else:
            motor_stop()

        sleep(0.05)

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    print("Cleaning up GPIO resources...")
    pwm_motor.stop()
    GPIO.cleanup()
    print("Done!")
