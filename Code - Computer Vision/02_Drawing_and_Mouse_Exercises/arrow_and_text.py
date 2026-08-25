# Lesson: draw an arrowed line + text label onto a loaded image.
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

img = cv.imread('../assets/test2.png', 0) # 1 color | 0 gray | -1 alpha
# blank = np.zeros((1024,1024,3),dtype='uint8') #blank canvas | 3 channel | 500x500 dimension
# blank2 = np.full((1024,1024,3),(255,255,255),dtype='uint8') # fill the canvas with white color | 3 channel | 500x500 dimension | colors
cv.arrowedLine(img,(0,img.shape[0]//2),(img.shape[1]//3,img.shape[0]//2),(255,250,255),thickness=20)
cv.putText(img,'genius',(img.shape[0]//2,img.shape[1]//2),cv.FONT_HERSHEY_COMPLEX,1.0,(255,255,0), thickness=2)
cv.imshow('line',img)

cv.waitKey(0) # press 0 to destroy window
cv.destroyAllWindows



