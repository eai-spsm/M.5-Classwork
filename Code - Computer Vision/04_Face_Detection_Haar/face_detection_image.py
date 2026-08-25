# Lesson: Haar face detection on a static high-res photo (rescaled down first).
import cv2 as cv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img = cv.imread(os.path.join(BASE_DIR, '..', 'assets', 'IMG_7847.JPG'))

def rescaleFrame(frame,scale=0.3):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

frame_resized = rescaleFrame(img)

xml_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Initialize the Face Cascade Classifier
face_cascde = cv.CascadeClassifier(xml_path)
gray_img = cv.cvtColor(frame_resized, cv.COLOR_RGB2GRAY)

# Parameters for face detection:
# scale: How much the image size is reduced at each image scale
# minNeighbor: How many neighbors each candidate rectangle should have to retain it
scale = 1.9
minNeighbor = 3
face_detect = face_cascde.detectMultiScale(gray_img, scale, minNeighbor)
if len(face_detect) > 0:
    for x, y, w, h in face_detect:
        cv.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("No faces detected.")

cv.imshow("Face detect",frame_resized)
cv.waitKey(0)
cv.destroyAllWindows()