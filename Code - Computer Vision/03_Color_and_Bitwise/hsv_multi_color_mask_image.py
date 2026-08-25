# Lesson: HSV masking for 3 colors at once (white/yellow/green), combined with bitwise_or.
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

# White mask
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])
white_mask = cv.inRange(hsv,lower_white, upper_white)

# Yellow mask
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([32, 255, 255])
mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

# Green mask
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
green_mask = cv.inRange(hsv, lower_green, upper_green)

pre = cv.bitwise_or(mask_yellow, green_mask)
pre2 = cv.bitwise_or(pre, white_mask)

result = cv.bitwise_and(test, test, mask=pre2)
cv.imshow('Original', test)
cv.imshow('detect color mask', pre2)
cv.imshow('Result', result)
cv.waitKey(0)
cv.destroyAllWindows()