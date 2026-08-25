# W7b Exercise: YOLO Webcam Detection -> 16x2 I2C LCD
# Same LCD wiring/module as w7a.py, plus a webcam running YOLO object
# detection — the LCD shows whatever class names are currently detected.
#
# Wiring:
#   LCD SDA -> GPIO2 (pin 3), LCD SCL -> GPIO3 (pin 5)  (same as w7a.py)
#   USB webcam plugged in (shows up as /dev/video0)
#
# One-time Pi setup before this will run (same as w7a.py):
#   sudo raspi-config          # Interface Options -> I2C -> Enable
#   sudo reboot
#   pip install rpi_lcd        # inside your venv
#
# Needs best.pt (custom YOLO model, already in this repo) and a display
# for the cv.imshow() window (see README "SSH / remote access").

import cv2 as cv
from ultralytics import YOLO
from rpi_lcd import LCD

MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.5
IMG_SIZE = 320
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
LCD_COLS = 16

model = YOLO(MODEL_PATH)
lcd = LCD()
lcd.clear()

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

last_detected = set()
print("Starting video feed. Press 'q' or Ctrl+C to quit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame from camera.")
            break

        results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        boxes = results[0].boxes

        detected = set()
        for box in boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected.add(name)

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 5, 0)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if detected != last_detected:
            lcd.clear()
            if detected:
                names = ", ".join(sorted(detected))[:LCD_COLS]
                lcd.text("Detected:", 1)
                lcd.text(names, 2)
            else:
                lcd.text("Scanning...", 1)
            last_detected = detected

        cv.imshow("YOLO Detection", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    cap.release()
    cv.destroyAllWindows()
    lcd.clear()
    print("Done!")
