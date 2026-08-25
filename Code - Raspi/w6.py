# W6 Exercise: MediaPipe Hand Tracking -> Servo Pointer
# Tracks the index fingertip via the webcam and maps its horizontal
# position to a servo angle (0-180 degrees), so the servo "points"
# toward wherever the finger is in frame.
#
# Wiring (BCM numbering):
#   Servo signal (orange/yellow) -> GPIO18
#   Servo power (red)            -> 5V (use an external 5V supply if the
#                                   servo draws more current than the Pi can spare)
#   Servo ground (brown/black)   -> Pi GND
#
# Needs hand_landmarker.task in the same folder (already in this repo).

import time
import cv2 as cv
import mediapipe as mp
import RPi.GPIO as GPIO

SERVO_PIN = 18
MODEL_PATH = "hand_landmarker.task"
INDEX_FINGERTIP = 8

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

cap = cv.VideoCapture(0)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

start_time = time.time()
print("Point your index finger at the camera. Press 'q' to quit.")

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame.")
                break

            frame = cv.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                tip = result.hand_landmarks[0][INDEX_FINGERTIP]
                x, y = int(tip.x * frame_w), int(tip.y * frame_h)

                angle = (x / frame_w) * 180
                duty = 2 + (angle / 18.0)
                servo.ChangeDutyCycle(duty)

                cv.circle(frame, (x, y), 8, (0, 255, 0), 2)
                cv.putText(frame, f"{angle:.0f} deg", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv.imshow("Hand Tracking Servo Pointer", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    cap.release()
    cv.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
    print("Done!")
