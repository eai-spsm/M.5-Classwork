# Lesson: click-drag-release to draw a right-angle triangle (hypotenuse/opposite/adjacent, color-coded).
import cv2 as cv
import numpy as np

# 1. Load the image and verify it exists
canvas = cv.imread('../assets/test2.png', 1)
t = 'Click to draw'
points = []
drawing = False

# 2. Define the callback function
def on_mouse(event, x, y, flags, param):
    global initx, inity, canvas, drawing
    
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True  # Start drawing state
        initx, inity = x, y 
        points.append((x, y)) 
        cv.circle(canvas, (initx, inity), 6, (0, 255, 0), thickness=-1)
        cv.imshow(t, canvas)                                        
        
    elif event == cv.EVENT_MOUSEMOVE:
        # Only show the preview line if the user is actively dragging
        if drawing:
            temp_canvas = canvas.copy()
            currentx, currenty = x, y
            cv.line(temp_canvas, (initx, inity), (currentx, currenty), (0, 255, 0), thickness=2)
            cv.imshow(t, temp_canvas)                           
            
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False  # Stop drawing state to block mousemove logic
        finalx, finaly = x, y
        points.append((finalx, finaly))
        cv.circle(canvas, (finalx, finaly), 6, (0, 255, 0), thickness=-1)
        
        # --- ALL TRIANGLE LOGIC MUST STAY WITHIN THIS BLOCK ---
        cornerA = (initx, inity)
        cornerB = (finalx, finaly)
        cornerC = (initx, finaly) # The right-angle corner point
        
        # Draw the 3 lines of the triangle permanently on the canvas
        cv.line(canvas, cornerA, cornerB, (0, 255, 0), thickness=2)  # Hypotenuse (Green)
        cv.line(canvas, cornerA, cornerC, (0, 0, 255), thickness=2)  # Opposite (Red)
        cv.line(canvas, cornerB, cornerC, (255, 0, 0), thickness=2)  # Adjacent (Blue)
        
        # Refresh the main display with the drawn triangle
        cv.imshow(t, canvas)

# 3. Setup the window and event listener
cv.namedWindow(t)                               
cv.imshow(t, canvas)                               
cv.setMouseCallback(t, on_mouse)             
cv.waitKey(0)                                   
cv.destroyAllWindows()
