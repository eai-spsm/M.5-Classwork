# Lesson: same right-angle triangle drawing, but each of the 3 sides gets a random color per draw.
import cv2 as cv
import numpy as np

# 1. Load the image and verify it exists
canvas = cv.imread('../assets/test2.png', 1)
t = 'Click to draw'
points = []  # Avoid naming variables 'list' as it's a Python built-in keyword
drawing = False
colors = [(0, 255, 0)] * 3  # [hypotenuse, opposite, adjacent]

def rand_color():
    return tuple(int(v) for v in np.random.randint(0, 256, 3))

# 2. Define the callback function
def on_mouse(event, x, y, flags, param):
    global initx, inity, canvas, drawing, colors
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        initx, inity = x, y
        colors = [rand_color(), rand_color(), rand_color()]
        points.append((x, y))
        cv.circle(canvas, (initx, inity), 6, colors[0], thickness=-1)
        cv.imshow(t, canvas)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            temp_canvas = canvas.copy()
            currentx, currenty = x, y
            cv.line(temp_canvas, (initx, inity), (currentx, currenty), colors[0], thickness=2)
            cv.imshow(t, temp_canvas)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        finalx, finaly = x, y
        points.append((finalx, finaly))
        cv.circle(canvas, (finalx, finaly), 6, colors[0], thickness=-1)

        # --- ALL TRIANGLE LOGIC MUST STAY WITHIN THIS BLOCK ---
        cornerA = (initx, inity)
        cornerB = (finalx, finaly)
        cornerC = (initx, finaly) # The right-angle corner point

        # Draw the 3 lines of the triangle permanently on the canvas
        cv.line(canvas, cornerA, cornerB, colors[0], thickness=2)  # Hypotenuse
        cv.line(canvas, cornerA, cornerC, colors[1], thickness=2)  # Opposite
        cv.line(canvas, cornerB, cornerC, colors[2], thickness=2)  # Adjacent
        
        # Refresh the main display with the drawn triangle
        cv.imshow(t, canvas)
        

# 3. Setup the window and event listener
cv.namedWindow(t)                               
cv.imshow(t, canvas)                               
cv.setMouseCallback(t, on_mouse)             
cv.waitKey(0)                                   
cv.destroyAllWindows()