# Lesson: extends hand_tracking_optimized_color_sample.py -- matches the sampled color to a target palette and points a Raspberry Pi servo at the matching angle. Needs RPi.GPIO (Pi only).
import cv2 as cv
import mediapipe as mp
import numpy as np
import RPi.GPIO as GPIO

OFFSET = (15, 15)  # pixels offset (dx, dy) from fingertip to sample color at
MATCH_THRESHOLD = 60  # max RGB distance to accept a color match
SERVO_PIN = 18

# Reference colors (R, G, B) -> servo angle on the semicircle wheel.
# Angle order follows the wheel top-to-bottom: yellow(180) ... green(22.5).
COLOR_TARGETS = [
    ("yellow",     (240, 210, 60),  180.0),
    ("steel_blue", (70, 120, 180),  157.5),
    ("black",      (40, 40, 40),    135.0),
    ("orange",     (230, 150, 70),  112.5),
    ("sky_blue",   (70, 150, 210),   90.0),
    ("pink",       (220, 100, 160),  67.5),
    ("coral",      (230, 110, 90),   45.0),
    ("green",      (90, 170, 80),    22.5),
]

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz, standard SG90
servo.start(0)

def set_servo_angle(angle):
    duty = 2 + (angle / 18)  # 0deg->2%, 180deg->12% duty cycle
    servo.ChangeDutyCycle(duty)

def match_color(r, g, b):
    best_name, best_angle, best_dist = None, None, float("inf")
    for name, (cr, cg, cb), angle in COLOR_TARGETS:
        dist = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
        if dist < best_dist:
            best_name, best_angle, best_dist = name, angle, dist
    if best_dist <= MATCH_THRESHOLD:
        return best_name, best_angle
    return None, None

# 1. Setup MediaPipe Tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,               # Tracking 2 hands is much faster than 4
    min_detection_confidence=0.5,  # Detection threshold
    min_tracking_confidence=0.5   # Tracking threshold
) as hands:

    # Frame skipping variables
    frame_count = 0
    SKIP_FACTOR = 2  # Process 1 frame, skip the next N frames (2 = process every 3rd frame)
    last_results = None  # Cache results to draw on skipped frames
    last_color = None  # avoid re-issuing the same servo command every frame

    while cap.isOpened():
        ret, frame = cap.read()
        h, w, c = frame.shape
        if not ret:
            break
        frame = cv.flip(frame, 1)

        if frame_count % (SKIP_FACTOR + 1) == 0:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            last_results = hands.process(rgb_frame)
        
        if last_results and last_results.multi_hand_landmarks:
            for hand_landmarks in last_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                tip = hand_landmarks.landmark[8]
                x8, y8 = int(tip.x * w), int(tip.y * h)

                sx = min(max(x8 + OFFSET[0], 0), w - 1)
                sy = min(max(y8 + OFFSET[1], 0), h - 1)
                b, g, r = frame[sy, sx]

                cv.circle(frame, (sx, sy), 5, (0, 255, 0), 2)
                swatch = np.full((100, 100, 3), (b, g, r), dtype=np.uint8)
                cv.imshow('Sampled Color', swatch)

                name, angle = match_color(int(r), int(g), int(b))
                if name is not None and name != last_color:
                    set_servo_angle(angle)
                    last_color = name
                    print(f"pointing at {name}, servo -> {angle} deg")

        frame_count += 1

        cv.imshow('Optimized Hand Tracking', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
servo.stop()
GPIO.cleanup()