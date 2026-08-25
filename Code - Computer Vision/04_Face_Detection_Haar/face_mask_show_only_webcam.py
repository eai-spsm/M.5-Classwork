# Lesson: Haar face detection + ellipse mask to SHOW ONLY the face region on webcam, blank out the rest.
import cv2 as cv
import numpy as np
import os

xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')
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

    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("face masking", result)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
