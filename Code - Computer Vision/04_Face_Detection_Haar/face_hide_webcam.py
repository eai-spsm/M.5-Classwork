# Lesson: Haar face detection + ellipse mask to BLANK OUT (hide) the face region on webcam, keep everything else.
import cv2 as cv
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier(xml_path)

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for x, y, w, h in faces:
        cv.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)

    result = cv.bitwise_and(frame, frame, mask=cv.bitwise_not(mask))

    cv.imshow("face hide", result)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
