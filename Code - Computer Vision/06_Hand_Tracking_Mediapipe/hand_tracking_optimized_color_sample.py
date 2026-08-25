# Lesson: hand tracking with frame-skipping for speed, samples the pixel color near the index fingertip and shows it as a swatch.
import cv2 as cv
import mediapipe as mp
import numpy as np

OFFSET = (15, 15)  # pixels offset (dx, dy) from fingertip to sample color at

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

        frame_count += 1

        cv.imshow('Optimized Hand Tracking', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()