# Lesson: fill a convex polygon (diamond) on a blank canvas with cv2.fillConvexPoly.
import cv2
import numpy as np

# 1. Create a blank black image (400x400 pixels, 3 color channels)
img = np.zeros((400, 400, 3), dtype=np.uint8)

# 2. Define the vertices of a convex polygon (e.g., a diamond/pentagon shape)
# Note: The array must have a shape of (N, 2) or (N, 1, 2) with dtype np.int32
pts = np.array([
    [200, 50],   # Top
    [350, 150],  # Right top
    [300, 320],  # Right bottom
    [100, 320],  # Left bottom
    [50, 150]    # Left top
], dtype=np.int32)

# 3. Parameters
color = (0, 255, 0)   # Green color in BGR format
line_type = cv2.LINE_AA # Anti-aliased line for smoother edges

# 4. Fill the convex polygon
cv2.fillConvexPoly(img, pts, color, lineType=line_type)

# 5. Display the result
cv2.imshow("Convex Polygon", img)
cv2.waitKey(0)
cv2.destroyAllWindows()