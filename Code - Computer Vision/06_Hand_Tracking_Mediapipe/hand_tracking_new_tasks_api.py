# Lesson: current mediapipe Tasks API (HandLandmarker) replacement for the deprecated mp.solutions.hands used in hand_tracking_basic_solutions_api.py.
"""
Hand tracking with MediaPipe Tasks API (new API).

Note: MediaPipe 0.10.30+ / 1.x removed mp.solutions.
Use HandLandmarker instead of mp.solutions.hands.
"""
import os
import time
import urllib.request
 
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 
# Connections between the 21 hand landmarks (for drawing skeleton)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),            # palm
]
 
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
 
 
def ensure_model(path: str = MODEL_PATH) -> str:
    """Download the hand landmarker model if missing."""
    if not os.path.exists(path):
        print(f"Downloading model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Done.")
    return path
 
 
def draw_hand_landmarks(img, landmarks):
    """Draw 21 landmarks + bones with OpenCV (no mp.solutions needed)."""
    h, w = img.shape[:2]
    points = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
 
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, points[a], points[b], (255, 0, 0), 2)
 
 
def main():
    model_path = ensure_model()
 
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
 
    try:
        while True:
            success, img = cap.read()
            if not success:
                break
 
            # OpenCV = BGR → convert to RGB for MediaPipe
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
 
            # VIDEO mode needs a monotonically increasing timestamp (ms)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
 
            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    draw_hand_landmarks(img, hand_lms)
 
            cv2.imshow("Hand Tracking", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 