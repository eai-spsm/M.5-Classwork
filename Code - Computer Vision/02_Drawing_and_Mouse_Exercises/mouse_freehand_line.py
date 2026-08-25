# Lesson: left-click repeatedly to drop connected dots, drawing a freehand line.
import cv2 as cv
import numpy as np

# 1. Load the image and verify it exists
canvas = cv.imread('../assets/test2.png', 1)
t = 'Click to draw'
dot = []
def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global initx, inity, drawing, canvas
        initx, inity = x, y
        dot.append((x, y))  
        cv.circle(canvas, (x, y), 6, (0, 255, 0), thickness=-1)
        cv.imshow(t, canvas)                   
        cv.waitKey(1)
    if len(dot) > 1:
        cv.line(canvas, dot[-2], dot[-1], (0, 255, 0), thickness=2)
        cv.imshow(t, canvas) # Refresh the window to show the new line
cv.namedWindow(t)                               
cv.imshow(t, canvas)                      
cv.setMouseCallback(t, on_mouse)             
cv.waitKey(0)                                   
cv.destroyAllWindows()