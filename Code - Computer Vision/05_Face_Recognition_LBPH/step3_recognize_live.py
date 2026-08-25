# Lesson step 3/3: load classifier.xml (from step2) and recognize faces live on webcam, labeled by name + confidence.
import cv2 as cv
import os
from datetime import datetime
from step2_train_recognizer import NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, 'classifier.xml')

CONFIDENCE_THRESHOLD = 70  # LBPH distance; higher = looser match, lower = stricter

face_cascade = cv.CascadeClassifier(XML_PATH)
clf = cv.face.LBPHFaceRecognizer_create()
clf.read(MODEL_PATH)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        id, con = clf.predict(face_img)
        name = NAMES.get(id, "unknown") if con <= CONFIDENCE_THRESHOLD else "unknown"
        label = f"{name} ({id}) conf={con:.1f}"
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.putText(frame, "Image Processing", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv.putText(frame, timestamp, (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.imshow("Face Recognition Test", frame)
    if cv.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
