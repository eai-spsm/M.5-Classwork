# W5b Exercise: DC Motor Driver — Terminal Speed Control
# Same wiring as w5a, but controlled by typed commands instead of buttons:
#   F/B/S = forward / backward / stop, +/- = speed up/down by 10%
#
# Wiring (BCM numbering):
#   Driver ENA -> GPIO13, IN1 -> GPIO18, IN2 -> GPIO12
#   Driver GND -> Pi GND (shared ground with motor supply)
#   Motor power -> separate motor supply, NOT the Pi's 5V pin

import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1 = 18
IN2 = 12
ENA = 13

GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

pwm_motor = GPIO.PWM(ENA, 1000)
pwm_motor.start(0)

current_speed = 40
current_direction = "STOP"


def motor_forward(speed):
    global current_direction
    current_direction = "F"
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm_motor.ChangeDutyCycle(speed)


def motor_backward(speed):
    global current_direction
    current_direction = "B"
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm_motor.ChangeDutyCycle(speed)


def motor_stop():
    global current_direction
    current_direction = "STOP"
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwm_motor.ChangeDutyCycle(0)


try:
    print("--- Terminal Motor Controller Active ---")
    print("Commands: F=forward  B=backward  S=stop  +=speed up  -=speed down")

    while True:
        print(f"\n[Status] Direction: {current_direction} | Speed: {current_speed}%")
        user_input = input("Enter command (+, -, F, B, S): ").strip().upper()

        if user_input == "+":
            current_speed = min(current_speed + 10, 100)
            if current_direction == "F":
                motor_forward(current_speed)
            elif current_direction == "B":
                motor_backward(current_speed)

        elif user_input == "-":
            current_speed = max(current_speed - 10, 0)
            if current_direction == "F":
                motor_forward(current_speed)
            elif current_direction == "B":
                motor_backward(current_speed)

        elif user_input == "F":
            print("Driving Forward...")
            motor_forward(current_speed)

        elif user_input == "B":
            print("Driving Backward...")
            motor_backward(current_speed)

        elif user_input == "S":
            print("Stopping Motor...")
            motor_stop()

        else:
            print("Invalid command. Please use +, -, F, B, or S.")

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    print("Cleaning up GPIO resources...")
    pwm_motor.stop()
    GPIO.cleanup()
    print("Done!")
