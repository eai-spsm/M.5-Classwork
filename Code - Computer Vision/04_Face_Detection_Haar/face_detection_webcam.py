# Lesson: Haar face detection running live on the webcam feed.
import cv2 as cv
import os
cap = cv.VideoCapture(0)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
face_cascde = cv.CascadeClassifier(xml_path)

while (cap.isOpened()):
    chack, frame = cap.read()
    
    if chack:
        gray_cap = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        face_detect = face_cascde.detectMultiScale(gray_cap)
        if len(face_detect) > 0:
            for x, y, w, h in face_detect:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            print("No faces detected.")
        
        cv.imshow("face detect", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if no frame is received (end of video or camera error)
        break

# Remember to release the capture and close windows when done
cap.release()
cv.destroyAllWindows()