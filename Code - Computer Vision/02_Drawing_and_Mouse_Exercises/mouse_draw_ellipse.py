# Lesson: click-drag-release to draw an ellipse, with a live white preview while dragging.
from datetime import datetime
import cv2 as cv
import numpy as np

# Global variables to track state
drawing = False
start_point = None
end_point = None

# Main canvas where finalized ellipses are kept
canvas = np.zeros((500, 700, 3), dtype=np.uint8)
# Temporary image used to render live preview while dragging
display_img = canvas.copy()


def draw_ellipse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, canvas, display_img

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)
        # Refresh display image from main canvas so old preview lines disappear
        display_img = canvas.copy()

        cx = (start_point[0] + end_point[0]) // 2
        cy = (start_point[1] + end_point[1]) // 2
        center = (cx, cy)

        axes_x = abs(end_point[0] - start_point[0]) // 2
        axes_y = abs(end_point[1] - start_point[1]) // 2

        # Draw real-time preview (white thin line)
        if axes_x > 1 or axes_y > 1:
            cv.ellipse(
                display_img, center, (axes_x, axes_y), 0, 0, 360, (255, 255, 255), 2
            )

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        if start_point and end_point:
            cx = (start_point[0] + end_point[0]) // 2
            cy = (start_point[1] + end_point[1]) // 2
            center = (cx, cy)

            axes_x = abs(end_point[0] - start_point[0]) // 2
            axes_y = abs(end_point[1] - start_point[1]) // 2

            if axes_x > 1 or axes_y > 1:
                # Commit the final colored ellipse to the main canvas
                color = tuple(np.random.randint(0, 256, 3).tolist())
                thickness = 6
                cv.ellipse(
                    canvas, center, (axes_x, axes_y), 0, 0, 360, color, thickness
                )

        # Update display image with final canvas state
        display_img = canvas.copy()


# Window setup
cv.namedWindow("Ellipse Drawing")
cv.setMouseCallback("Ellipse Drawing", draw_ellipse_callback)

# Main display loop
while True:
    # Render display_img instead of canvas so preview is visible
    cv.imshow("Ellipse Drawing", display_img)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:  # Press 'q' or Esc to exit
        break

cv.destroyAllWindows()