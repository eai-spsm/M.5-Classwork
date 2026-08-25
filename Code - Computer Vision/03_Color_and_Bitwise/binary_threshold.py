# Lesson: cv.threshold binary thresholding on a grayscale screenshot.
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

# Binary Image Thresholding
img = cv.imread('../assets/Screenshot 2026-06-24 124535.png', 0)

# cv.threshold returns two values: retval and dst. 
# We use '_' to discard the threshold value and keep the image.
_, result = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

cv.imshow("hehe", result) 
cv.waitKey(0)
cv.destroyAllWindows()