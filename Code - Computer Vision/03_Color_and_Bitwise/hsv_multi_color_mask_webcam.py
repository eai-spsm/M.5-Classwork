# Lesson: same 3-color HSV mask (white/yellow/green) as hsv_multi_color_mask_image.py, applied live to webcam frames.
import cv2 as cv
import numpy as np


capture = cv.VideoCapture(0)
while(capture.isOpened()): # check if the video is opened | SAFER 
    ret, frame = capture.read() # ret is boolean value to check if the frame is read correctly, frame is the actual frame
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
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
        result = cv.bitwise_and(frame, frame, mask=pre2)
          
        cv.imshow('Frame', result)
        key =  cv.waitKey(33)
        if key == ord('d'): # if press d to destroy window
             break
        if cv.waitKey(25) & 0xFF == ord('d'): # wait key is how many sec u transition
            break
    else:
        break
    
capture.release() 
cv.destroyAllWindows()