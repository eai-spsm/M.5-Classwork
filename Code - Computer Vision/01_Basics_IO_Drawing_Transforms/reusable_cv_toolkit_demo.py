# Lesson: usage demo for reusable_cv_toolkit.py -- run this file to see every
# toolkit function fired once, or copy a numbered block into your own script.
"""
============================================================================
 reusable_cv_toolkit.py -- DEMO / USAGE EXAMPLES
============================================================================
Each numbered block is a self-contained demo of the toolkit's functions.
Run the whole file (each block pops its own window -- press any key, or 'd'
in video, to move on), or copy one block into your own script.

    import reusable_cv_toolkit_demo   # runs everything
    -- or just copy a block + this import line:
    from reusable_cv_toolkit import *
============================================================================
"""

from reusable_cv_toolkit import (
    SAMPLES,
    load, save, show, wait, destroy, plot, set_mouse,
    video, set_res, play,
    rescale, resize, scale_matrix,
    crop, translate, rotate, rotate90, flip,
    gray, hsv, lab, rgb, convert,
    split, merge, channel,
    blank, blank_like, rectangle, circle, line, text,
    blur, canny, sobel_x, sobel_y, sobel, laplacian,
    threshold, threshold_otsu, adaptive,
    dilate, erode, find_contours, draw_contours,
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
)
import cv2 as cv


# ============================================================================
# 1. Show an image
# ============================================================================
img = load(SAMPLES.img)            # flag 1 color | 0 gray | -1 alpha
show(img, '1. Original')
wait()
destroy()


# ============================================================================
# 2. Play a video (or webcam) -- press 'd' to quit
# ============================================================================
play(SAMPLES.video)                # or play(SAMPLES.WEBCAM)


# ============================================================================
# 3. Resize -- three ways
# ============================================================================
img = load(SAMPLES.img)
show(rescale(img, 0.5), '3a. Rescale 0.5x')          # by factor
show(resize(img, 500, 500), '3b. Resize 500x500')    # exact (W, H)
show(scale_matrix(img, 0.5, 0.5), '3c. Scale matrix') # affine matrix diagonal
wait()
destroy()


# ============================================================================
# 4. Transforms -- crop, translate, rotate, flip
# ============================================================================
img = load(SAMPLES.img)
show(crop(img, 200, 50, 400, 200), '4a. Crop')        # x1, y1, x2, y2
show(translate(img, 100, 100), '4b. Translate')       # +x right, +y down
show(rotate(img, 45), '4c. Rotate 45')                # CCW degrees
show(flip(img, 0), '4d. Flip vertical')               # 0 vert | 1 horiz | -1 both
wait()
destroy()


# ============================================================================
# 5. Color spaces
# ============================================================================
img = load(SAMPLES.img)
show(gray(img), '5a. Gray')
show(hsv(img), '5b. HSV')
show(lab(img), '5c. LAB')
wait()
destroy()
plot(img, '5d. matplotlib (auto BGR->RGB)')           # correct colors in plt


# ============================================================================
# 6. Draw shapes & text on a blank canvas
# ============================================================================
canvas = blank((500, 500, 3))             # (H, W, 3) black canvas
rectangle(canvas, (50, 50), (200, 200), color=(0, 255, 0), thickness=2)
circle(canvas, (350, 150), radius=60, color=(0, 0, 255), thickness=-1)
line(canvas, (0, 0), (500, 500), color=(255, 255, 255), thickness=3)
text(canvas, 'preset', org=(120, 420), scale=1.5)
show(canvas, '6. Drawing')
wait()
destroy()


# ============================================================================
# 7. Vision pipeline -- gray -> blur -> canny -> contours
# ============================================================================
img = rescale(load(SAMPLES.img), 0.5)
edges = canny(blur(gray(img)), 125, 175)              # tweak thresholds here
contours, _ = find_contours(edges)
print(f'7. {len(contours)} contour(s) found')

canvas = blank_like(img)                               # 3-channel black canvas
draw_contours(canvas, contours, color=(0, 255, 0), thickness=2)

show(edges, '7a. Canny edges')
show(canvas, '7b. Contours')
wait()
destroy()


# ============================================================================
# 8. Threshold variants + dilate / erode
# ============================================================================
g = gray(load(SAMPLES.img))
show(threshold(g, 150), '8a. Threshold binary')
show(threshold_otsu(g), '8b. Threshold Otsu')
show(adaptive(g, block=11, c=3), '8c. Adaptive')

e = canny(g)
show(dilate(e, ksize=3, iterations=2), '8d. Dilated (2 iters)')
show(erode(dilate(e), ksize=3, iterations=1), '8e. Eroded')
wait()
destroy()


# ============================================================================
# 9. Color channels -- split, view each in color, merge back
# ============================================================================
img = load(SAMPLES.img)
b, g, r = split(img)                          # 3 grayscale channels
show(b, '9a. Blue (gray)')
show(channel(img, 0), '9b. Blue (in color)')  # 0 B | 1 G | 2 R
show(channel(img, 2), '9c. Red (in color)')
show(merge([b, g, r]), '9d. Merged back')
wait()
destroy()


# ============================================================================
# 10. Bitwise -- and / or / xor / not  (on two shapes)
# ============================================================================
rect = rectangle(blank((400, 400)), (30, 30), (370, 370), color=255, thickness=-1)
circ = circle(blank((400, 400)), (200, 200), radius=200, color=255, thickness=-1)
show(bitwise_and(rect, circ), '10a. AND')     # overlap only
show(bitwise_or(rect, circ), '10b. OR')       # both combined
show(bitwise_xor(rect, circ), '10c. XOR')     # non-overlap
show(bitwise_not(rect), '10d. NOT')           # inverted
wait()
destroy()


# ============================================================================
# 11. Gradient edge detection -- Sobel & Laplacian
# ============================================================================
g = gray(load(SAMPLES.img))
show(sobel_x(g), '11a. Sobel X (vertical edges)')
show(sobel_y(g), '11b. Sobel Y (horizontal edges)')
show(sobel(g), '11c. Sobel combined')
show(laplacian(g), '11d. Laplacian')
wait()
destroy()


# ============================================================================
# 12. Mouse click events -- left-click to drop a dot
# ============================================================================
canvas = load(SAMPLES.img)
WIN = '12. Click to draw (left-click)'

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:          # left button pressed
        circle(canvas, (x, y), radius=6, color=(0, 255, 0), thickness=-1)
        show(canvas, WIN)
        print(f'click at ({x}, {y})')

show(canvas, WIN)
set_mouse(WIN, on_mouse)
wait()                                          # keeps window alive for events
destroy()


# ============================================================================
# 13. Live video pipeline -- run canny on every frame ('d' to quit)
# ============================================================================
play(SAMPLES.video, process=lambda f: canny(gray(rescale(f, 0.5))),
     title='13. Live Canny')
