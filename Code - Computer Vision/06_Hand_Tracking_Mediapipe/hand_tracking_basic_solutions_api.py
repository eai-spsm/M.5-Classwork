# Lesson: hand skeleton tracking via mediapipe's old mp.solutions.hands API (deprecated in mediapipe 0.10.30+/1.x -- see hand_tracking_new_tasks_api.py for the replacement).
import cv2 as cv
import mediapipe as mp

# 1. Initialize Hand Tracking and Drawing tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

def rescaleFrame(frame, scale):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)
    return cv.resize(frame,dimension, interpolation=cv.INTER_AREA)

# 2. Configure Hands Model
with mp_hands.Hands(
    static_image_mode=False,       # Set False for video streaming
    max_num_hands=4,               # Detect up to 2 hands
    min_detection_confidence=0.5,  # Detection threshold
    min_tracking_confidence=0.5   # Tracking threshold
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = rescaleFrame(frame, 0.4)
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame) # process to get result

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv.imshow('MediaPipe Hand Tracking', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()