# Lesson: live webcam feed with a circle/line/rectangle overlay + a live HH:MM:SS clock.
import cv2 as cv
from datetime import datetime

capture = cv.VideoCapture(0)

while capture.isOpened(): 
    ret, frame = capture.read()  
    if ret == True:
        h, w = frame.shape[0], frame.shape[1]
        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")

        # Shapes and overlays
        center_x = w // 2
        center_y = h // 2
        cv.circle(frame, (center_x, center_y), 50, (0, 255, 0), thickness=-1) 
        
        cv.line(frame, (0, 0), (w, h), (0, 0, 255), 3)
        cv.rectangle(frame, (10, 10), (w - 10, h - 10), (255, 191, 0), thickness=20)
        
        # Draw the live time string on top of the background
        cv.putText(
            frame, 
            f'live: {time_string}', 
            (40, 63),                  # Coordinates (X, Y) of the bottom-left corner of the text
            cv.FONT_HERSHEY_SIMPLEX,   # Font style
            0.8,                       # Font scale (size)
            (0,0,0),           # Font color (White)
            2,                         # Thickness of the strokes
            cv.LINE_AA                 # Anti-aliased line for smoother rendering
        )
        # -------------------------
        
        cv.imshow('Frame', frame)
        if cv.waitKey(33) & 0xFF == ord('d'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()