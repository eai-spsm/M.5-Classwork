# Lesson: click-drag-release with the mouse to draw a rectangle, live preview while dragging.
import cv2 as cv
import numpy as np
import random

img = np.zeros((600,600,3), dtype=np.uint8)
drawing = False
ix, iy = -1, -1
ex, ey = -1, -1
current_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
def click_position(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, current_color

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        ex, ey = x, y
        current_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)) 

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            ex, ey = x, y  

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        cv.rectangle(img, (ix, iy), (ex, ey), current_color, 2)

cv.namedWindow("homework")
cv.setMouseCallback("homework", click_position)

while True:
    display_img = img.copy() 
    if drawing:
        cv.rectangle(display_img, (ix, iy), (ex, ey), current_color, 2)
    cv.imshow("homework", display_img)

    if cv.waitKey(10) == 27:  
        break

cv.destroyAllWindows()
