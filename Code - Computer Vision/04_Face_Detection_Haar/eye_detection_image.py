# Lesson: Haar eye detection on a static photo.
import cv2 as cv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img = cv.imread(os.path.join(BASE_DIR, '..', 'assets', '97941_0.jpg'))

xml_path = os.path.join(BASE_DIR, 'haarcascade_eye_tree_eyeglasses.xml')
eye_cascde = cv.CascadeClassifier(xml_path)

gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

scale = 1.1
minNeighbor = 10
eye_detect = eye_cascde.detectMultiScale(gray_img, scale, minNeighbor)

print(eye_detect)
for (x, y, w, h) in eye_detect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

cv.imshow("eye detect", img)
cv.waitKey(0)
cv.destroyAllWindows()
