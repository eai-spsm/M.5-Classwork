# Lesson: HSV in-range masking to isolate green pixels in a static image.
import cv2 as cv
from datetime import datetime
import numpy as np
img = cv.imread('../assets/col.jpg')

def rescaleFrame(frame, scale):
    # .shape = height[0], width[1], layers/channel [2]
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)
    return cv.resize(frame,dimension, interpolation=cv.INTER_AREA)

test = rescaleFrame(img, 0.5)
hsv = cv.cvtColor(test, cv.COLOR_BGR2HSV)

# Green mask
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
green_mask = cv.inRange(hsv, lower_green, upper_green)

result = cv.bitwise_and(test, test, mask=green_mask)
cv.imshow('Original', test)
cv.imshow('detect color mask', green_mask)
cv.imshow('Result', result)
cv.waitKey(0)
cv.destroyAllWindows()