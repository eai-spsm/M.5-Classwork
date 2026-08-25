# Lesson: click a pixel to read its BGR value and show it as text + a solid color swatch.
import cv2 as cv
import numpy as np
img = cv.imread('../assets/test2.png', 1)
def click_position(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        blue = img[y,x,0]
        green = img[y,x,1]
        red = img[y,x,2]
        text = f'({x}, {y}) BGR: ({blue}, {green}, {red})'
        cv.putText(img,text,(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv.imshow('image',img)
        blank = np.zeros((500,500,3),dtype='uint8') #blank canvas | 3 channels | 500x500 dimension
        blank[:] = [blue, green, red]  # Fill the canvas with the color of the pixel
        cv.imshow('blank',blank)

cv.imshow('image',img)
cv.setMouseCallback('image',click_position)
cv.waitKey(0)
cv.destroyAllWindows()