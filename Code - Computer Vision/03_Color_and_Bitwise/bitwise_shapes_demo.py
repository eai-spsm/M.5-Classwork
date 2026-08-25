# Lesson: AND/OR/XOR/NOT on two same-size shapes (filled rectangle vs filled circle).
import cv2 as cv
import numpy as np


rect = np.zeros((400, 400), dtype='uint8')
cv.rectangle(rect, (30, 30), (370, 370), 255, thickness=-1)    # filled white box

circle = np.zeros((400, 400), dtype='uint8')
cv.circle(circle, (200, 200), 200, 255, thickness=-1)          # filled white circle

cv.imshow('AND (overlap)', cv.bitwise_and(rect, circle))       # where BOTH white
cv.imshow('OR (both)', cv.bitwise_or(rect, circle))            # where EITHER white
cv.imshow('XOR (differ)', cv.bitwise_xor(rect, circle))        # non-overlap
cv.imshow('NOT (inverted)', cv.bitwise_not(rect))              # white <-> black

cv.waitKey(0)
cv.destroyAllWindows()