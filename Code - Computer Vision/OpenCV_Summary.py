"""
============================================================================
 OpenCV / Computer Vision -- FULL COURSE SUMMARY (basics -> YOLO + MediaPipe)
============================================================================
One reference file spanning the whole "Code - Computer Vision" folder, in
the order you'd actually learn it. Each numbered section below corresponds
to one (or several) lesson file(s) in the numbered topic folders next to
this file -- read the section here for the concept + every argument
explained, then open the matching folder for the runnable exercise version.

Grouping in this version:
  1. Setup & Paths
  2. Reading & Displaying Images
  3. Reading & Displaying Video
  4. Resizing / Rescaling Frames (for video / live use)
  5. Changing Capture Resolution
  6. Drawing Shapes & Text                          -> 02_Drawing_and_Mouse_Exercises/
  7. SIMPLE CONTROLS  -- resize, crop, translate, rotate, flip
                         (basic one-shot transforms on a single image)
  8. EDGE DETECTION + CONTOURS -- the full "vision pipeline":
                         grayscale -> blur -> canny/threshold -> dilate/erode
                         -> findContours -> drawContours
  9. COLOR SPACES -- BGR <-> Grayscale / HSV / LAB / RGB, plus showing an
                         image correctly with matplotlib
 10. COLOR CHANNELS -- split an image into B/G/R, view each, merge back
 11. BITWISE -- AND / OR / XOR / NOT to combine images & masks   -> 03_Color_and_Bitwise/
 12. GRADIENT EDGES -- Sobel (x / y) and Laplacian edge detection
 13. MOUSE EVENTS -- setMouseCallback to react to clicks / movement
 14. FACE DETECTION (Haar Cascade) + MASKING a detected region   -> 04_Face_Detection_Haar/
 15. FACE RECOGNITION (LBPH) -- capture -> train -> recognize    -> 05_Face_Recognition_LBPH/
 16. HAND TRACKING (MediaPipe) -- landmarks, and driving a servo -> 06_Hand_Tracking_Mediapipe/
 17. YOLO OBJECT DETECTION -- load a model, run inference, train -> 07_YOLO_Object_Detection/

HOW TO READ THIS FILE:
  - Each command has a `#~ function(...)` block explaining EVERY argument.
  - Right under it is a `# Example:` set -- the simplest copy-paste usage
    (kept short & commented so reading/running the file isn't interrupted).
  - The runnable demo lines below the example actually pop windows.

This file is meant to be READ section by section, not run top to bottom --
several sections call cv.imshow() + cv.waitKey(0), which BLOCKS until you
press a key / close the window. Every cv.imshow() title below has been
renamed to describe WHAT THAT STEP DID, so the popup windows are
self-explanatory. Sections 15-17 are reference/pseudocode (no built-in
webcam/dataset available at import time) -- run the actual lesson files in
their numbered folders instead.
============================================================================
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt   # used in Section 9 to display RGB correctly

# Paths used throughout this file (this script lives in "Code - Computer Vision",
# shared images/video sit in the assets/ folder next to it)
IMG_PATH = 'assets/img.jpg'
VIDEO_PATH = 'assets/Sequence 01.mp4'


# ============================================================================
# WARNING -- TWO GOTCHAS THAT BITE EVERY OPENCV BEGINNER
# ============================================================================
# 1) COORDINATE ORDER: (x,y) vs (row,col) -- NOT the same order!
#    - NumPy array indexing / slicing  -> img[row, col] = img[y, x]  (y FIRST)
#      e.g. cropping:  img[y1:y2, x1:x2]
#    - img.shape                       -> (height, width, channels) = (rows, cols, ch)
#    - Drawing/geometry functions (cv.rectangle, cv.circle, cv.line,
#      cv.putText, mouse callback x/y, cv.resize's dsize) ALL take (x, y)
#      -- x (column/horizontal) FIRST, y (row/vertical) SECOND.
#    Mixing these up (indexing with x,y or drawing with y,x) is the single
#    most common OpenCV bug -- see Section 4 for the resize-specific case.
#
# 2) COLOR ORDER: OpenCV is BGR, not RGB -- and HSV ranges are NOT 0-360/0-100.
#    - cv.imread / cv.imshow / all cv.* drawing colors use (B, G, R) order.
#      Handing a BGR array to matplotlib (which expects RGB) swaps red/blue
#      (see Section 9) -- always cv.cvtColor(img, cv.COLOR_BGR2RGB) first.
#    - cv.cvtColor(img, cv.COLOR_BGR2HSV) does NOT give the "usual" HSV
#      ranges. OpenCV packs it into uint8, so:
#          H (hue)        : 0-179   (NOT 0-360 -- actual hue halved to fit a byte)
#          S (saturation) : 0-255   (NOT 0-100)
#          V (value)      : 0-255   (NOT 0-100)
#      Color-picking with an external tool (e.g. a 0-360 hue wheel)? Divide
#      that hue by 2 before using it as an OpenCV H threshold.
# ============================================================================


# ============================================================================
# SECTION 2: READING & DISPLAYING IMAGES
# ============================================================================
#~ cv.imread(path, flag)
#   path : string path to the image file (relative to wherever you RUN the
#          script from)
#   flag : controls HOW the file is loaded into the array
#       1  -> cv.IMREAD_COLOR     : load as 3-channel color, in BGR order
#                                    (NOT RGB! red and blue channels are swapped)
#       0  -> cv.IMREAD_GRAYSCALE : load as a single-channel grayscale image
#       -1 -> cv.IMREAD_UNCHANGED : load exactly as stored, including the
#                                    alpha (transparency) channel if present
#
#~ cv.imshow(window_name, img)
#   window_name : string title shown on the popup window's title bar.
#                  Re-using the same name updates that window in place.
#   img         : the NumPy array (image) to display
#
#~ cv.waitKey(delay)
#   delay = 0   -> wait FOREVER until any key is pressed (returns key code)
#   delay = N   -> wait up to N milliseconds, then continue automatically
#   The return value is an int; mask with `& 0xFF` before comparing to
#   `ord('x')` because some platforms return extra bits in higher positions.
#
#~ cv.destroyAllWindows()
#   Closes every OpenCV window. MUST be called with parentheses `()` to
#   actually execute -- writing `cv.destroyAllWindows` (no parens) just
#   references the function and does nothing.
#
# Example:
#   img  = cv.imread(IMG_PATH, 1)     # 1 color (BGR) | 0 gray | -1 keep alpha
#   cv.imshow('window title', img)    # pop up a window showing the image
#   cv.waitKey(0)                     # 0 = wait forever for any key press
#   cv.destroyAllWindows()            # close every OpenCV window

img = cv.imread(IMG_PATH, 1)        # 1 = color (BGR)
cv.imshow('Original Image', img)
cv.waitKey(0)
cv.destroyAllWindows()


# ============================================================================
# SECTION 3: READING & DISPLAYING VIDEO
# ============================================================================
#~ cv.VideoCapture(source)
#   source : EITHER a string path to a video file, OR an integer camera
#            index (0 = default webcam, 1 = next camera, etc.)
#
#~ capture.read()
#   Grabs the next frame. Returns a tuple (isTrue, frame):
#       isTrue -> bool, True if a frame was successfully grabbed,
#                 False once the video ends (or the source is invalid)
#       frame  -> the frame as a NumPy array, same shape/format as an image
#
#~ capture.release()
#   Frees the file/camera handle. Always call this when you're done so
#   other programs (or a later cv.VideoCapture call) can use the device.
#
# COMMON ERROR: "error -215:Assertion failed ... !_src.empty()"
#   Almost always means the path passed to VideoCapture()/imread() does
#   NOT exist, so OpenCV got an empty/None frame and choked on it.
#
# Example:
#   capture = cv.VideoCapture(VIDEO_PATH)   # file path ... or 0 for webcam
#   while True:
#       isTrue, frame = capture.read()      # grab the next frame
#       if not isTrue:                      # no frame -> video ended
#           break
#       cv.imshow('Video', frame)
#       if cv.waitKey(20) & 0xFF == ord('d'):   # press 'd' to quit early
#           break
#   capture.release()                       # free the handle when done

capture = cv.VideoCapture(VIDEO_PATH)

while True:
    isTrue, frame = capture.read()         # grab next frame
    if not isTrue:                         # video ended / read failed -> stop
        break

    cv.imshow('Video Playback', frame)

    # cv.waitKey(20)              -> wait 20ms between frames (~50 fps)
    # & 0xFF                      -> mask to the lowest 8 bits (cross-platform safety)
    # == ord('d')                 -> ord() converts 'd' to its ASCII code (100)
    if cv.waitKey(20) & 0xFF == ord('d'):  # press 'd' to quit early
        break

capture.release()
cv.destroyAllWindows()


# ============================================================================
# SECTION 4: RESIZING / RESCALING FRAMES (images AND video)
# ============================================================================
# IMPORTANT -- AXIS ORDER GOTCHA:
#   img.shape  -> (height, width, channels)   <- rows first, then columns
#   cv.resize(src, dsize, ...) wants dsize = (width, height)  <- REVERSED!
# Mixing these two orders up is one of THE most common OpenCV bugs.
#
# There are two common ways to resize:
#
#   METHOD 1 -- SCALE FACTOR
#     Multiply the current width/height by a ratio (e.g. 0.5 = half size,
#     2.0 = double size). Aspect ratio is preserved automatically. Great
#     for shrinking video frames so they fit on screen / process faster.
#
#   METHOD 2 -- HARDCODED TARGET DIMENSIONS
#     Tell cv.resize() the exact (width, height) you want. Simple, but the
#     image will look stretched/squashed if that ratio doesn't match the
#     original. (Used in Section 7, "Simple Controls".)

# --- METHOD 1: scale factor ---
def rescaleFrame(frame, scale):
    """
    Return a copy of `frame` resized by `scale`.
    scale = 0.5 -> half size, scale = 2.0 -> double size.
    Works for images AND individual video frames (both are just arrays).
    """
    # NEW SYNTAX: frame.shape returns a tuple (height, width, channels).
    # frame.shape[1] -> index 1 of that tuple -> WIDTH
    # frame.shape[0] -> index 0 of that tuple -> HEIGHT
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)          # cv.resize wants (width, height)

    #~ cv.resize(src, dsize, interpolation)
    #   src           : the image/frame to resize
    #   dsize         : target size as (width, height)
    #   interpolation : algorithm used to fill in / drop pixels (Section 7 cheat sheet)
    #                    INTER_AREA is the recommended choice for SHRINKING
    #
    # Example:
    #   half = cv.resize(img, None, fx=0.5, fy=0.5)                     # by factor
    #   exact = cv.resize(img, (640, 480), interpolation=cv.INTER_AREA) # to (W, H)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# --- Example: apply Method 1 to a video ---
capture = cv.VideoCapture(VIDEO_PATH)
while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame_resized = rescaleFrame(frame, scale=0.5)
    cv.imshow('Video Playback (Original)', frame)
    cv.imshow('Video Playback (Rescaled 0.5x)', frame_resized)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()

# --- Example: apply Method 1 to a still image ---
img = cv.imread(IMG_PATH, 1)
resized_img = rescaleFrame(img, scale=0.5)
cv.imshow('Image Rescaled 0.5x', resized_img)
cv.waitKey(0)
cv.destroyAllWindows()


# ============================================================================
# SECTION 5: CHANGING CAPTURE RESOLUTION (live video / webcam)
# ============================================================================
# rescaleFrame() resizes a frame AFTER it has been captured.
# capture.set(...) instead changes the resolution the CAMERA captures at.
# Works reliably on live webcams; video FILES often ignore it.
#
#~ capture.set(propId, value) / capture.get(propId)
#   propId : an integer ID (or the equivalent cv.CAP_PROP_* constant --
#            they are interchangeable, the constant is just a friendly name)
#   value  : the value to set that property to
#
# Example:
#   capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)   # set capture width  (id 3)
#   capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)   # set capture height (id 4)
#   fps = capture.get(cv.CAP_PROP_FPS)           # read a property back (id 5)
#
# --- Capture property cheat sheet (most useful ones) ---
#   ID | Constant                     | What it controls
#   ---|------------------------------|----------------------------------
#   3  | cv.CAP_PROP_FRAME_WIDTH      | Frame width
#   4  | cv.CAP_PROP_FRAME_HEIGHT     | Frame height
#   5  | cv.CAP_PROP_FPS              | Frame rate (frames per second)
#   7  | cv.CAP_PROP_FRAME_COUNT      | Total frame count (video files only)
#   10 | cv.CAP_PROP_BRIGHTNESS       | Camera brightness
#   11 | cv.CAP_PROP_CONTRAST         | Camera contrast
#   12 | cv.CAP_PROP_SATURATION       | Camera saturation
#   15 | cv.CAP_PROP_EXPOSURE         | Camera exposure (often negative)
#   39 | cv.CAP_PROP_FOCUS            | Camera hardware focus
#   40 | cv.CAP_PROP_AUTO_EXPOSURE    | Auto exposure (0 = manual, 1 = auto)

def changeRes(capture, width, height):
    """Set a LIVE VideoCapture's resolution (id 3 = width, id 4 = height)."""
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width)    # same as capture.set(3, width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)  # same as capture.set(4, height)

# Example (uncomment if a webcam is connected):
# capture = cv.VideoCapture(0)
# changeRes(capture, 1280, 720)


# ============================================================================
# SECTION 6: DRAWING SHAPES & WRITING TEXT
# ============================================================================
# np.zeros(shape, dtype) -- creates a blank canvas filled with 0s (black).
#
# NEW SYNTAX -- shape vs shape[:2]:
#   img.shape     -> (height, width, channels)  e.g. (500, 500, 3)
#   img.shape[:2] -> SLICE of the tuple, keeping only index 0 and 1
#                    -> (height, width)  e.g. (500, 500)   (channels dropped)
#   `[:2]` means "everything from the start UP TO (not including) index 2".
#
# QUICK SUMMARY (when to use which):
#   np.zeros(img.shape,      dtype='uint8') -> 3-channel BLACK image
#                                               (for color drawing/painting)
#   np.zeros(img.shape[:2],  dtype='uint8') -> 1-channel BLACK image
#                                               (for masks / stencils / contours)
#   np.zeros((500, 500, 3),  dtype='uint8') -> manual 500x500 3-channel canvas
#
# Example:
#   blank = np.zeros((500, 500, 3), dtype='uint8')   # black 3-channel canvas
#   blank[:] = 0, 255, 0                              # paint it all green (B,G,R)
blank = np.zeros((500, 500, 3), dtype='uint8')   # 500x500 black BGR canvas
cv.imshow('Blank Canvas', blank)

# blank[:] = (B, G, R)  -- slice-assign paints EVERY pixel that color.
# Remember: OpenCV channel order is BLUE, GREEN, RED (not RGB!)
blank[:] = 0, 255, 0   # pure green
cv.imshow('Canvas Painted Green', blank)

#~ cv.rectangle(img, pt1, pt2, color, thickness)
#   pt1       : (x, y) of the TOP-LEFT corner
#   pt2       : (x, y) of the BOTTOM-RIGHT corner
#   color     : (B, G, R) tuple
#   thickness : outline width in pixels, OR cv.FILLED (== -1) to fill solid
#
# Example:
#   cv.rectangle(img, (0, 0), (250, 250), (0, 255, 0), thickness=2)   # 2px outline
#   cv.rectangle(img, (0, 0), (250, 250), (0, 255, 0), thickness=-1)  # filled
cv.rectangle(blank, (0, 0), (250, 250), (0, 250, 0), thickness=2)              # 2px outline
cv.rectangle(blank, (0, 0), (250, 250), (0, 250, 0), thickness=cv.FILLED)      # filled
# Fill just the top-left quadrant of the canvas using its own dimensions:
cv.rectangle(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (0, 250, 0), thickness=cv.FILLED)
cv.imshow('Rectangle Drawn', blank)

#~ cv.circle(img, center, radius, color, thickness)
#   center    : (x, y) of the circle's center point
#   radius    : radius in pixels (a single int)
#   thickness : outline width, or -1 / cv.FILLED to fill
#
# Example:
#   cv.circle(img, (250, 250), 40, (0, 0, 255), thickness=-1)   # filled red dot
cv.circle(blank, (250, 250), 40, (0, 250, 0), thickness=-1)
cv.imshow('Circle Drawn', blank)

#~ cv.line(img, pt1, pt2, color, thickness)
#   pt1, pt2 : (x, y) endpoints of the line
#
# Example:
#   cv.line(img, (0, 0), (300, 400), (255, 255, 255), thickness=3)   # white line
cv.line(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (255, 250, 255), thickness=3)
cv.line(blank, (0, 0), (300, 400), (255, 250, 255), thickness=3)
cv.imshow('Line Drawn', blank)

#~ cv.putText(img, text, org, fontFace, fontScale, color, thickness)
#   text      : the string to draw
#   org       : (x, y) of the BOTTOM-LEFT corner of the text
#   fontFace  : a font constant, e.g. cv.FONT_HERSHEY_COMPLEX
#   fontScale : size multiplier (1.0 = default size)
#
# Example:
#   cv.putText(img, 'hello', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
cv.putText(blank, 'hello', (255, 255), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), thickness=2)
cv.imshow('Text Drawn', blank)

cv.waitKey(0)
cv.destroyAllWindows()


# ============================================================================
# SECTION 7: SIMPLE CONTROLS -- Resize, Crop, Translate, Rotate, Flip
# ============================================================================
# These are the basic "one-shot" transforms you apply to a single image --
# as opposed to Section 4 (rescaling, mainly for video/live frames).

img = cv.imread(IMG_PATH, 1)

# --- Resize (hardcoded target dimensions) ---
#~ cv.resize(src, dsize, interpolation)
#   dsize         : (width, height) -- REMEMBER: opposite order from .shape!
#   interpolation : algorithm used when the size changes (cheat sheet below)
#
#   Goal                      | Flag                | Notes
#   --------------------------|---------------------|----------------------------
#   Shrinking                 | cv.INTER_AREA       | Avoids jagged edges/moire
#   Enlarging (best quality)  | cv.INTER_CUBIC      | Sharp & smooth, slower
#   Enlarging (fast)          | cv.INTER_LINEAR     | Default, good speed/quality
#   Pixel art / AI masks      | cv.INTER_NEAREST    | No blur, crisp blocky pixels
#   Max quality zoom          | cv.INTER_LANCZOS4   | 8x8 neighborhood, slowest
#
# Example:
#   small = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)   # shrink
#   big   = cv.resize(img, (1500, 1500), interpolation=cv.INTER_CUBIC) # enlarge
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
cv.imshow('Resized (500x500, INTER_AREA)', resized)

# --- Resize via an affine MATRIX (the np.float32 way) ---
# You can also scale with cv.warpAffine + a 2x3 matrix, just like translate/
# rotate below. The DIAGONAL of the matrix holds the x / y scale factors:
#
#       [[sx, 0,  0],     sx -> horizontal scale (x-axis)
#        [0,  sy, 0]]     sy -> vertical scale   (y-axis)
#
#   - REDUCE the diagonal numbers (e.g. 0.5) to SHRINK the image.
#   - RAISE them (e.g. 2.0) to ENLARGE it.
#   - sx != sy stretches/squashes (e.g. 2.0, 1.0 = twice as wide, same tall).
#   The 3rd column (0, 0) is the x/y SHIFT -- set it for translate (Section
#   below); keep it 0 for a pure resize.
#
# cv.resize() is the simpler/faster choice for plain scaling -- this matrix
# form is mainly useful when you want to scale AND shift/rotate in one step.
#
# Example:
#   mat = np.float32([[0.5, 0, 0],
#                     [0, 0.5, 0]])           # shrink to 50%
#   scaled = cv.warpAffine(img, mat, (img.shape[1], img.shape[0]))   # (W, H) out
scale_mat = np.float32([[0.5, 0, 0],          # 0.5 on the diagonal = half size
                        [0, 0.5, 0]])
scaled = cv.warpAffine(img, scale_mat, (img.shape[1], img.shape[0]))
cv.imshow('Resized via Affine Matrix (0.5x)', scaled)

# --- Crop ---
# Images are just NumPy arrays, so cropping is plain ARRAY SLICING:
#
#   cropped = img[y1:y2, x1:x2]
#       - 1st range (y1:y2) -> ROWS    -> the VERTICAL   (y) range
#       - 2nd range (x1:x2) -> COLUMNS -> the HORIZONTAL (x) range
#       - (0, 0) is the TOP-LEFT corner of the image
#
# img[50:200, 200:400] keeps rows 50-200 and columns 200-400
# -> a region that is 150px tall and 200px wide.
#
# Example:
#   cropped = img[50:200, 200:400]   # [y1:y2, x1:x2] -- the box to KEEP
cropped = img[50:200, 200:400]
cv.imshow('Cropped Region', cropped)

# --- Translate (shift) ---
#~ cv.warpAffine(src, M, dsize)
# Applies a 2x3 "affine transformation matrix" M to every pixel coordinate.
#   src   : input image
#   M     : 2x3 transformation matrix (np.float32)
#   dsize : output size as (width, height)
def translate(img, x, y):
    """
    Shift `img` by (x, y) pixels.
    x > 0 -> right, x < 0 -> left
    y > 0 -> down,  y < 0 -> up
    """
    # [[1, 0, x],    -> new_x = 1*old_x + 0*old_y + x
    #  [0, 1, y]]    -> new_y = 0*old_x + 1*old_y + y
    # i.e. just adds (x, y) to every coordinate -- a pure shift.
    transMat = np.float32([[1, 0, x], [0, 1, y]])  # warpAffine needs float32
    dimensions = (img.shape[1], img.shape[0])      # (width, height)
    return cv.warpAffine(img, transMat, dimensions)

# Example:
#   translated = translate(img, 100, 100)   # right 100px, down 100px
#   translated = translate(img, -50, -30)   # left  50px, up   30px
translated = translate(img, 100, 100)  # right 100px, down 100px
cv.imshow('Translated Image', translated)

# --- Rotate ---
def rotate(img, angle, rotPoint=None):
    """
    Rotate `img` by `angle` degrees (counter-clockwise for positive values)
    around `rotPoint`. If rotPoint is None, rotates around the image center.
    """
    # NEW SYNTAX: img.shape[:2] -> slice of the shape tuple -> (height, width)
    # (drops the channel count, since we don't need it for rotation math)
    (height, width) = img.shape[:2]

    if rotPoint is None:              # default arg -> caller can omit it
        rotPoint = (width // 2, height // 2)  # center of the image

    #~ cv.getRotationMatrix2D(center, angle, scale)
    #   center : (x, y) pivot point to rotate around
    #   angle  : degrees, counter-clockwise for positive values
    #   scale  : 1.0 = no extra resizing, 0.5 = also shrink to half, etc.
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)

# Example:
#   rotated = rotate(img, 45)                   # 45 deg around the center
#   rotated = rotate(img, -90)                  # 90 deg clockwise
#   fast90  = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)   # built-in 90 deg turns
rotated = rotate(img, 45)  # rotate 45 degrees around the center
cv.imshow('Rotated 45 Degrees', rotated)

# --- Flip ---
#~ cv.flip(src, flipCode)
#   flipCode = 0  -> flip VERTICALLY   (upside down)
#   flipCode = 1  -> flip HORIZONTALLY (mirror left-right)
#   flipCode = -1 -> flip BOTH axes    (180 degree mirror)
#
# Example:
#   cv.flip(img, 0)    # vertical (upside down)
#   cv.flip(img, 1)    # horizontal (mirror)
#   cv.flip(img, -1)   # both axes
flip = cv.flip(img, 0)
cv.imshow('Flipped Vertically', flip)

cv.waitKey(0)
cv.destroyAllWindows()


# ============================================================================
# SECTION 8: EDGE DETECTION + CONTOURS -- the "Vision Pipeline"
# ============================================================================
# === THE VISION PIPELINE LAYER ===
# 1. Grayscale + Blur  -> clean, single-channel input (less noise)
# 2. cv.threshold()    -> BINARIZES: converts to clean, solid black & white blocks
#    cv.Canny()        -> PLOTS EDGES: scans and highlights individual boundary pixels
# 3. cv.dilate/erode   -> thicken/thin those edges
# 4. cv.findContours() -> CONNECTS THE DOTS: links edge/boundary pixels into an
#                         organized list of shapes (contours)
# 5. cv.drawContours() -> draw the result

img = cv.imread(IMG_PATH, 1)

# --- Step 1: Grayscale ---
#~ cv.cvtColor(src, code)
#   code : a cv.COLOR_<FROM>2<TO> constant describing the conversion
#       cv.COLOR_BGR2GRAY -> BGR color  -> single-channel grayscale
#       cv.COLOR_BGR2RGB  -> BGR color  -> RGB (needed before matplotlib)
#       cv.COLOR_BGR2HSV  -> BGR color  -> Hue/Saturation/Value
#
# Example:
#   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   # color -> grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray)

# --- Step 1b: Blur ---
#~ cv.GaussianBlur(src, ksize, sigmaX)
#   ksize  : kernel ("window") size as (width, height). BOTH numbers must
#            be POSITIVE and ODD, e.g. (3,3), (5,5), (7,7). Bigger = blurrier.
#   sigmaX : standard deviation of the gaussian curve in the X direction.
#            cv.BORDER_DEFAULT here is just a numeric placeholder that lets
#            OpenCV auto-compute sigma from ksize.
# Blurring removes small noise that would otherwise create lots of tiny,
# useless edges/contours.
#
# Example:
#   blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)   # 5x5 = mild blur
#   blur = cv.GaussianBlur(gray, (9, 9), cv.BORDER_DEFAULT)   # 9x9 = heavier
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Gaussian Blur', blur)

# --- Step 2a: Canny edge detection ---
#~ cv.Canny(src, threshold1, threshold2)
#   threshold1, threshold2 : the LOW and HIGH gradient-strength thresholds.
#       - A pixel with edge-strength ABOVE threshold2 is DEFINITELY an edge.
#       - A pixel BELOW threshold1 is DEFINITELY NOT an edge.
#       - Anything in between is kept ONLY if connected to a strong edge.
#       Lower the thresholds to detect MORE (fainter) edges; raise them to
#       detect FEWER, only-strong edges.
#   Returns a binary (black/white) image: white = edge pixel.
#
# Example:
#   canny = cv.Canny(img, 125, 175)    # straight off the original
#   canny = cv.Canny(blur, 125, 175)   # blur first -> fewer noisy edges
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edge Detection', canny)

# --- Step 2b: Threshold (alternative / complementary to Canny) ---
#~ cv.threshold(src, thresh, maxval, type)
#   src    : a SINGLE-CHANNEL (grayscale) image
#   thresh : the cutoff value (0-255) -- "what to put here" depends on how
#            bright/dark your subject is vs. the background. Common starting
#            point: 125-150, then tweak.
#   maxval : the value given to pixels that PASS the test (almost always 255)
#   type   : the comparison rule applied to each pixel:
#       cv.THRESH_BINARY      -> pixel > thresh ? maxval : 0
#       cv.THRESH_BINARY_INV  -> pixel > thresh ? 0 : maxval        (inverted)
#       cv.THRESH_TRUNC       -> pixel > thresh ? thresh : pixel    (clip/cap)
#       cv.THRESH_TOZERO      -> pixel > thresh ? pixel : 0
#       cv.THRESH_TOZERO_INV  -> pixel > thresh ? 0 : pixel
#
#   WORKED EXAMPLE -- thresh=150, maxval=255, type=THRESH_BINARY:
#       Every pixel is checked ONE AT A TIME against `thresh`:
#           input pixel : 10    90    150   151   200   255
#           is it > 150?: No    No    No    Yes   Yes   Yes
#           output      : 0     0     0     255   255   255
#       So `dst` ends up containing ONLY two values (0 or 255) -- that's
#       why it's called "binary": each pixel is now either fully OFF
#       (black) or fully ON (white), with nothing in between.
#   Returns a TUPLE: (retval, dst)
#       retval -> the threshold value that was actually used
#       dst    -> the resulting binary image
#
# Example:
#   ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)       # standard
#   ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)   # inverted
ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Threshold (Binary)', thresh)

# --- Note: OTSU (6th type) + adaptive thresholding ---
# cv.THRESH_OTSU is a 6th type, OR'd onto one of the 5 above -- it AUTO-picks
# `thresh` for you (good when lighting is fairly even). `thresh` arg is
# ignored when OTSU is used, so pass 0:
#~ cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#
#~ cv.adaptiveThreshold(src, maxval, adaptiveMethod, thresholdType, blockSize, C)
#   No single global `thresh` -- instead computes a LOCAL threshold per
#   neighborhood, which handles UNEVEN lighting / shadows much better.
#       adaptiveMethod : cv.ADAPTIVE_THRESH_MEAN_C or cv.ADAPTIVE_THRESH_GAUSSIAN_C
#       blockSize      : size of each neighborhood (odd number, e.g. 11)
#       C              : constant subtracted from the computed mean (fine-tune)
#
# Example:
#   ret, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#   adaptive  = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv.THRESH_BINARY, 11, 3)

# --- Step 3: Dilate / Erode (sharpen up the Canny edges) ---
#~ cv.dilate(src, kernel, iterations)
#   kernel     : shape/size of the neighborhood used to grow white regions,
#                e.g. (3,3) -- a 3x3 block
#   iterations : how many times to repeat the operation (more = thicker)
#
# Example:
#   dilated = cv.dilate(canny, (3, 3), iterations=1)   # thicken edges once
#   dilated = cv.dilate(canny, (3, 3), iterations=3)   # thicker (3 passes)
dilated = cv.dilate(canny, (3, 3), iterations=1)
cv.imshow('Dilated Edges', dilated)

#~ cv.erode(src, kernel, iterations)
#   The opposite of dilate -- shrinks white regions. Running erode after
#   dilate with the SAME kernel can "undo" some of the thickening, giving
#   thinner, more precise edge lines than the raw Canny output.
#
# Example:
#   eroded = cv.erode(dilated, (3, 3), iterations=1)   # thin the edges back
eroded = cv.erode(dilated, (3, 3), iterations=1)
cv.imshow('Eroded Edges', eroded)

# --- Step 4: Find contours ---
# A "contour" is the OUTLINE / boundary curve of a shape -- a list of (x, y)
# points tracing where a region of similar color/intensity meets a
# different one. Contours are NOT the same as edges: edges (Canny) are
# individual lines; contours are CLOSED SHAPES built by connecting those
# edges/boundaries. (its a list btw)
#
#~ cv.findContours(image, mode, method)
#   image  : a binary/edge image (single channel) -- here, canny or thresh
#   mode   : how contours are organized / which ones are returned
#       cv.RETR_LIST     -> ALL contours, no parent/child relationships
#       cv.RETR_EXTERNAL -> only the OUTERMOST contour of each shape
#       cv.RETR_TREE     -> ALL contours, with full nesting hierarchy
#       cv.RETR_CCOMP    -> all contours organized into a 2-level hierarchy
#   method : how much detail to keep PER contour
#       cv.CHAIN_APPROX_NONE   -> store EVERY point along the boundary
#                                 (most accurate, most memory)
#       cv.CHAIN_APPROX_SIMPLE -> compress straight line segments down to
#                                 just their START and END points (saves
#                                 memory, usually good enough)
#   Returns: (contours, hierarchy)
#       contours  : a tuple of arrays; each array is the (x, y) points for
#                   ONE contour (one shape found)
#       hierarchy : describes how contours are nested inside each other
#                   (only meaningful with RETR_TREE / RETR_CCOMP)
#
# You can find contours from EITHER `canny` or `thresh` -- canny is more
# common for general edge-based shapes.
#
# Example:
#   contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#   contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#   print(len(contours), 'shapes found')
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contour(s) found')

# --- Step 5: Draw contours ---
#~ cv.drawContours(image, contours, contourIdx, color, thickness)
#   image      : image to draw ON, modified IN PLACE -- usually a blank
#                canvas (np.zeros) or a copy of the original (img.copy())
#   contours   : the list returned by cv.findContours
#   contourIdx : which contour to draw -- an index into `contours`,
#                or -1 to draw ALL of them
#   color      : (B, G, R)
#   thickness  : outline thickness in pixels, or cv.FILLED to fill the shape
#
# NOTE: arguments must be separated by commas: (-1, (0, 0, 255), 2)
#       writing -1(0, 0, 255) -- with no comma -- is a syntax error, since
#       Python would try to "call" the int -1 like a function.
#
# Drawing in COLOR (0, 0, 255) needs a 3-CHANNEL canvas, so use
# np.zeros(img.shape, ...) here -- NOT img.shape[:2] (that's the 1-channel
# "mask/stencil" version from the Quick Summary above, which can only hold
# a single intensity value per pixel, not a B/G/R color).
#
# Example:
#   blank = np.zeros(img.shape, dtype='uint8')            # 3-channel canvas
#   cv.drawContours(blank, contours, -1, (0, 0, 255), 2)  # -1 = draw ALL, red
blank = np.zeros(img.shape, dtype='uint8')   # 3-channel canvas (for color drawing)
cv.drawContours(blank, contours, -1, (0, 0, 255), 2)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)
cv.destroyAllWindows()


# ============================================================================
# SECTION 9: COLOR SPACES -- BGR <-> Grayscale / HSV / LAB / RGB
# ============================================================================
# A "color space" is just a different way of encoding the SAME pixel colors.
# OpenCV loads images as BGR by default; converting to another space makes
# certain jobs easier (e.g. HSV for color-based thresholding/tracking).
#
#~ cv.cvtColor(src, code)
#   src  : the input image
#   code : a cv.COLOR_<FROM>2<TO> constant naming the conversion direction.
#          Conversions are REVERSIBLE -- there's a matching ..2BGR for each
#          (e.g. cv.COLOR_HSV2BGR) to convert back.
#
#   code               | Result / when to use it
#   -------------------|------------------------------------------------------
#   cv.COLOR_BGR2GRAY  | Single-channel grayscale (edges, contours, thresholds)
#   cv.COLOR_BGR2HSV   | Hue/Saturation/Value -- best for "find this color"
#   cv.COLOR_BGR2LAB   | Lightness + a/b -- perceptually even, good for compare
#   cv.COLOR_BGR2RGB   | RGB order -- needed before handing an image to matplotlib
#
# Example:
#   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   # grayscale
#   hsv  = cv.cvtColor(img, cv.COLOR_BGR2HSV)    # for color tracking
#   bgr  = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)    # convert back

img = cv.imread(IMG_PATH, 1)

# --- BGR -> Grayscale ---
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('BGR to Grayscale', gray)

# --- BGR -> HSV ---
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('BGR to HSV', hsv)

# --- BGR -> LAB ---
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('BGR to LAB', lab)

# --- BGR -> RGB ---
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('BGR to RGB', rgb)

cv.waitKey(0)
cv.destroyAllWindows()

# --- WHY matplotlib needs RGB ---
# matplotlib expects RGB, but OpenCV stores BGR. If you pass the raw OpenCV
# `img` straight to plt.imshow(), red and blue get SWAPPED (skin turns blue,
# sky turns orange, etc.). Convert with cv.COLOR_BGR2RGB first so the colors
# display correctly.
#
# Example:
#   plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))   # convert BEFORE plotting
#   plt.show()
plt.imshow(rgb)            # `rgb` is already converted above -> looks correct
plt.title('Shown via matplotlib (RGB)')
plt.axis('off')
plt.show()


# ============================================================================
# SECTION 10: COLOR CHANNELS -- split / merge
# ============================================================================
# A color image is 3 stacked 2D arrays (Blue, Green, Red). You can pull them
# apart, inspect/edit one, then stack them back together.
#
#~ cv.split(img)
#   Splits a 3-channel image into a tuple of 3 SINGLE-channel arrays (b, g, r).
#   Each one is grayscale -- it only stores HOW MUCH of that color is present,
#   so it shows up as a black-and-white intensity map (bright = lots of it).
#
#~ cv.merge(channels)
#   The inverse: takes a list [b, g, r] and stacks them back into one color
#   image. Order matters -- OpenCV expects Blue, Green, Red.
#
# Example:
#   b, g, r = cv.split(img)            # 3 grayscale channels
#   merged  = cv.merge([b, g, r])      # stack them back -> original image
#   blue    = cv.merge([b, blank, blank])   # see Blue in its OWN color

img = cv.imread(IMG_PATH, 1)
b, g, r = cv.split(img)
cv.imshow('Blue channel (gray)', b)
cv.imshow('Green channel (gray)', g)
cv.imshow('Red channel (gray)', r)

# To view a channel in ITS color, merge it with two BLANK (all-zero) channels
# so the other two colors contribute nothing.
blank = np.zeros(img.shape[:2], dtype='uint8')   # 1-channel black
cv.imshow('Blue (in color)', cv.merge([b, blank, blank]))   # B, G=0, R=0
cv.imshow('Green (in color)', cv.merge([blank, g, blank]))
cv.imshow('Red (in color)', cv.merge([blank, blank, r]))

# Merge the real channels back -> identical to the original image.
merged = cv.merge([b, g, r])
cv.imshow('Merged back to original', merged)

cv.waitKey(0)
cv.destroyAllWindows()


# ============================================================================
# SECTION 11: BITWISE OPERATIONS -- AND / OR / XOR / NOT
# ============================================================================
# Bitwise ops compare two images PIXEL BY PIXEL (and bit by bit). They are
# most useful with binary masks (black = 0, white = 255) to keep/remove
# regions of an image.
#
# IMPORTANT: both inputs must be the SAME SIZE (and usually same channels).
#
#~ cv.bitwise_and(src1, src2, mask=None) -> white only where BOTH are white  (intersection)
#~ cv.bitwise_or(src1, src2,  mask=None) -> white where EITHER is white      (union)
#~ cv.bitwise_xor(src1, src2, mask=None) -> white only where they DIFFER     (one but not both)
#~ cv.bitwise_not(src1,       mask=None) -> inverts every pixel (white <-> black)
#   mask : optional -- if given, the op is only applied where the mask is white.
#
# Example:
#   anded = cv.bitwise_and(img1, img2)     # keep pixels common to both
#   masked = cv.bitwise_and(img, img, mask=circle_mask)   # crop img to a shape
#   inverted = cv.bitwise_not(mask)        # flip a mask

# Two simple shapes to demonstrate (white on black, same 400x400 size):
rectangle = np.zeros((400, 400), dtype='uint8')
cv.rectangle(rectangle, (30, 30), (370, 370), 255, thickness=-1)   # filled square

circle = np.zeros((400, 400), dtype='uint8')
cv.circle(circle, (200, 200), 200, 255, thickness=-1)              # filled circle

cv.imshow('Source: Rectangle', rectangle)
cv.imshow('Source: Circle', circle)

cv.imshow('AND (intersection)', cv.bitwise_and(rectangle, circle))  # overlap only
cv.imshow('OR (union)', cv.bitwise_or(rectangle, circle))           # both shapes
cv.imshow('XOR (non-overlap)', cv.bitwise_xor(rectangle, circle))   # where they differ
cv.imshow('NOT (inverted)', cv.bitwise_not(rectangle))              # flipped

cv.waitKey(0)
cv.destroyAllWindows()


# ============================================================================
# SECTION 12: GRADIENT EDGES -- Sobel & Laplacian
# ============================================================================
# Canny (Section 8) is one edge detector; Sobel and Laplacian are the other
# classic ones. They measure the GRADIENT -- how fast pixel intensity changes
# -- which spikes at edges. Run them on a GRAYSCALE image.
#
# WHY CV_64F + convertScaleAbs:
#   Going dark->light is a POSITIVE gradient, light->dark is NEGATIVE. If the
#   output were uint8 (0-255) the negatives would be clipped to 0 and you'd
#   lose half your edges. So compute in cv.CV_64F (signed float), then
#   cv.convertScaleAbs() takes the ABSOLUTE value and casts back to uint8.
#
#~ cv.Sobel(src, ddepth, dx, dy, ksize)
#   ddepth : output depth -- use cv.CV_64F to keep negatives (see above)
#   dx, dy : order of the derivative per axis.
#            dx=1, dy=0 -> gradient along X -> highlights VERTICAL edges
#            dx=0, dy=1 -> gradient along Y -> highlights HORIZONTAL edges
#   ksize  : kernel size (odd, e.g. 3). Bigger = thicker, smoother edges.
#
#~ cv.Laplacian(src, ddepth, ksize)
#   Uses the 2nd derivative -- responds to edges in ALL directions at once
#   (no separate x/y). One call, no dx/dy needed.
#
# Example:
#   sx = cv.convertScaleAbs(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3))  # vertical
#   sy = cv.convertScaleAbs(cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3))  # horizontal
#   combined = cv.bitwise_or(sx, sy)                                    # both
#   lap = cv.convertScaleAbs(cv.Laplacian(gray, cv.CV_64F))            # all dirs

img = cv.imread(IMG_PATH, 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sobel_x = cv.convertScaleAbs(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3))   # vertical edges
sobel_y = cv.convertScaleAbs(cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3))   # horizontal edges
sobel_combined = cv.bitwise_or(sobel_x, sobel_y)
laplacian = cv.convertScaleAbs(cv.Laplacian(gray, cv.CV_64F))

cv.imshow('Sobel X (vertical edges)', sobel_x)
cv.imshow('Sobel Y (horizontal edges)', sobel_y)
cv.imshow('Sobel combined', sobel_combined)
cv.imshow('Laplacian', laplacian)

cv.waitKey(0)
cv.destroyAllWindows()


# ============================================================================
# SECTION 13: MOUSE EVENTS -- react to clicks / movement
# ============================================================================
# cv.setMouseCallback() attaches a FUNCTION that OpenCV calls every time the
# mouse does something over a named window (move, click, release, ...).
#
#~ cv.setMouseCallback(window_name, callback, param=None)
#   window_name : the title you passed to cv.imshow() -- the window MUST
#                 already exist before you attach the callback.
#   callback    : your function. Its signature is FIXED -- OpenCV always calls
#                 it with exactly these 5 args:
#
#       def on_mouse(event, x, y, flags, param):
#           event : which action happened (cv.EVENT_* constant, see below)
#           x, y  : mouse position in the window, in pixels
#           flags : bitmask of modifier keys / buttons held (Ctrl, Shift, ...)
#           param : the optional `param` you passed to setMouseCallback
#       (flags and param are often unused, but must stay in the signature.)
#
# --- Common event constants ---
#   cv.EVENT_MOUSEMOVE      mouse moved
#   cv.EVENT_LBUTTONDOWN    left button pressed
#   cv.EVENT_LBUTTONUP      left button released
#   cv.EVENT_RBUTTONDOWN    right button pressed
#   cv.EVENT_MBUTTONDOWN    middle button pressed
#   cv.EVENT_LBUTTONDBLCLK  left double-click
#
# Example:
#   def on_mouse(event, x, y, flags, param):
#       if event == cv.EVENT_LBUTTONDOWN:
#           print('clicked at', x, y)
#   cv.imshow('win', img)                 # 1) show the window first
#   cv.setMouseCallback('win', on_mouse)  # 2) then attach the callback
#   cv.waitKey(0)                          # 3) waitKey keeps it alive for events

canvas = cv.imread(IMG_PATH, 1)
WINDOW = 'Click to draw'

def on_mouse(event, x, y, flags, param):
    # Left-click -> drop a green dot where you clicked, and print the coords.
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(canvas, (x, y), 6, (0, 255, 0), thickness=-1)
        cv.imshow(WINDOW, canvas)         # redraw so the new dot appears
        print(f'click at ({x}, {y})')

cv.imshow(WINDOW, canvas)                 # window must exist before the callback
cv.setMouseCallback(WINDOW, on_mouse)
cv.waitKey(0)                             # blocks here, dispatching mouse events
cv.destroyAllWindows()


# ============================================================================
# SECTION 14: FACE DETECTION (Haar Cascade) + MASKING a region on video
# ============================================================================
# cv.CascadeClassifier loads a pretrained Haar cascade (an XML file of
# features) that scans a GRAYSCALE frame for a shape it was trained on
# (frontal faces here). Combine the detected box with a mask + bitwise_and
# to isolate (show-only) or hide a region, instead of just drawing a rectangle.
#
#~ cv.CascadeClassifier(xml_path).detectMultiScale(gray_img, scaleFactor, minNeighbors)
#   gray_img     : single-channel image -- cascades are trained on grayscale
#   scaleFactor  : how much the image is shrunk at each scale step (e.g. 1.3)
#   minNeighbors : how many overlapping detections are required to keep a hit
#                  (higher = fewer false positives, may miss real faces)
#   Returns a list of (x, y, w, h) boxes, one per detected face.
#
# --- Pattern: mask instead of just drawing a rectangle ---
#   1. Build a single-channel BLACK mask the same size as the frame
#      (np.zeros(frame.shape[:2], dtype=np.uint8))
#   2. Draw a FILLED shape (cv.ellipse / cv.rectangle, thickness=-1) onto the
#      mask at each detected face box -> mask is now WHITE where faces are
#   3. cv.bitwise_and(frame, frame, mask=mask)        -> SHOW ONLY the face(s)
#      cv.bitwise_and(frame, frame, mask=cv.bitwise_not(mask))  -> HIDE the face(s)
#
# Example:
#   face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#   gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#   faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#   mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#   for x, y, w, h in faces:
#       cv.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
#   show_only_face = cv.bitwise_and(frame, frame, mask=mask)
#   hide_face      = cv.bitwise_and(frame, frame, mask=cv.bitwise_not(mask))
#
# See runnable versions: 04_Face_Detection_Haar/face_mask_show_only_webcam.py
# (show only face) and 04_Face_Detection_Haar/face_hide_webcam.py (blank out
# the face, show everything else).


# ============================================================================
# SECTION 15: FACE RECOGNITION (LBPH) -- capture -> train -> recognize
# ============================================================================
# Haar (Section 14) answers "is there a face here?" -- LBPH (Local Binary
# Patterns Histograms) answers "WHOSE face is this?". It's a classic,
# lightweight recognizer (no deep learning / GPU needed) built into
# opencv-contrib as cv.face.LBPHFaceRecognizer_create().
#
# Three-step pipeline (see 05_Face_Recognition_LBPH/step1..3_*.py):
#   STEP 1 -- CAPTURE: detect a face with the same Haar cascade as Section
#             14, crop just the face region, save ~100 grayscale samples per
#             person as "user.<id>.<n>.jpg" in a data/ folder.
#   STEP 2 -- TRAIN: load every saved image, re-detect the face crop, collect
#             (face_image, numeric_id) pairs, then clf.train(faces, ids) and
#             clf.write('classifier.xml') to save the trained model.
#   STEP 3 -- RECOGNIZE: load classifier.xml, run the SAME Haar detection on
#             a live frame, then clf.predict(face_crop) for each detected
#             face to get back (id, confidence).
#
#~ cv.face.LBPHFaceRecognizer_create()
#   Requires opencv-contrib-python (the `cv.face` module isn't in the plain
#   opencv-python package).
#
#~ recognizer.train(faces, ids)
#   faces : list of grayscale face-crop arrays (all can be different sizes)
#   ids   : parallel list/array of integer labels, one per face
#
#~ recognizer.predict(face_img)
#   Returns (id, confidence). For LBPH, confidence is a DISTANCE -- LOWER
#   means a BETTER match (0 = perfect). This is the opposite of most ML
#   "confidence scores" where higher = better, so don't reuse that
#   intuition here. A common threshold is "confidence <= 70 -> known,
#   else unknown", but tune it against your own data.
#
# Example:
#   clf = cv.face.LBPHFaceRecognizer_create()
#   clf.train(faces, np.array(ids))          # STEP 2
#   clf.write('classifier.xml')
#
#   clf = cv.face.LBPHFaceRecognizer_create()
#   clf.read('classifier.xml')               # STEP 3
#   id, confidence = clf.predict(face_crop)
#   name = NAMES.get(id, 'unknown') if confidence <= 70 else 'unknown'
#
# See runnable versions: 05_Face_Recognition_LBPH/step1_capture_training_faces.py,
# step2_train_recognizer.py, step3_recognize_live.py.
# NOTE: the captured face crops (data/) and any real names mapped to IDs are
# biometric/personal data -- keep that folder out of version control
# (already covered by .gitignore in this repo).


# ============================================================================
# SECTION 16: HAND TRACKING (MediaPipe) -- landmarks + driving a servo
# ============================================================================
# MediaPipe's hand model returns 21 (x, y, z) landmarks per detected hand
# (wrist, then 4 joints per finger). Landmark 8 is the INDEX FINGERTIP --
# the most commonly used point for "pointing" interactions.
#
# TWO different mediapipe APIs exist across versions -- know which one your
# installed version uses:
#   OLD API (mediapipe < 0.10.30): mp.solutions.hands
#       mp_hands = mp.solutions.hands
#       with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
#                            min_tracking_confidence=0.5) as hands:
#           results = hands.process(rgb_frame)          # RGB in, not BGR!
#           for hand in results.multi_hand_landmarks:
#               tip = hand.landmark[8]                   # normalized 0-1 coords
#               x, y = int(tip.x * width), int(tip.y * height)
#
#   NEW API (mediapipe 0.10.30+ / 1.x): Tasks API, mp.solutions was removed
#       options = HandLandmarkerOptions(
#           base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
#           running_mode=VisionRunningMode.VIDEO, num_hands=2)
#       landmarker = HandLandmarker.create_from_options(options)
#       mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#       result = landmarker.detect_for_video(mp_image, timestamp_ms)  # ms must increase each call
#       tip = result.hand_landmarks[0][8]
#
# --- Common pattern: fingertip position -> action ---
#   1. Get normalized (x, y) in [0, 1] from the landmark.
#   2. Multiply by frame width/height to get pixel coordinates.
#   3. Map that position to whatever you're controlling:
#        - sample the pixel color underneath it (color picker)
#        - map x -> a servo angle (0-180) for a "pointer" (Raspberry Pi + GPIO.PWM)
#        - use it as a virtual cursor / drawing tool
#
# PERFORMANCE TIP -- frame skipping: running the model on every frame is
# slow on a CPU (e.g. Raspberry Pi). Process every 3rd frame and reuse the
# last result to draw on the skipped ones (see hand_tracking_optimized_*.py):
#   if frame_count % (SKIP_FACTOR + 1) == 0:
#       last_results = hands.process(rgb_frame)
#   # ... draw using last_results every frame regardless
#
# Example (old API, minimal):
#   mp_hands = mp.solutions.hands
#   mp_drawing = mp.solutions.drawing_utils
#   with mp_hands.Hands() as hands:
#       rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#       results = hands.process(rgb)
#       if results.multi_hand_landmarks:
#           for lm in results.multi_hand_landmarks:
#               mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
#
# See runnable versions in 06_Hand_Tracking_Mediapipe/:
#   hand_tracking_basic_solutions_api.py    -- old API, draws the skeleton
#   hand_tracking_new_tasks_api.py          -- new Tasks API equivalent
#   hand_tracking_optimized_color_sample.py -- frame-skipping + color sampling
#   hand_tracking_color_to_servo.py         -- samples color, points a Pi servo at a matched target


# ============================================================================
# SECTION 17: YOLO OBJECT DETECTION -- load, infer, train
# ============================================================================
# YOLO ("You Only Look Once") is a single-pass deep-learning object detector
# -- one forward pass through the network gives back EVERY object's box,
# class, and confidence at once (as opposed to Haar's sliding-window scan
# for one specific shape). Used here via the `ultralytics` package.
#
#~ YOLO(model_path)
#   model_path : a .pt weights file -- either a pretrained general model
#                (e.g. 'yolo11n.pt', downloads automatically) or your own
#                fine-tuned weights (e.g. 'best.pt')
#
#~ model(frame, conf=0.5, imgsz=320, verbose=False)
#   frame   : a single image / video frame (NumPy array, BGR is fine)
#   conf    : minimum confidence to keep a detection (0-1)
#   imgsz   : inference resolution -- SMALLER = faster but less accurate.
#             320 is a common choice for real-time CPU inference (e.g. Pi)
#   Returns a list of Results (one per input image); Results[0].boxes holds
#   every detected box for that frame.
#
#~ Reading detections out of a Results object
#   for box in results[0].boxes:
#       cls_id = int(box.cls[0])          # class index
#       name = model.names[cls_id]        # class index -> label string
#       conf = float(box.conf[0])         # confidence 0-1
#       x1, y1, x2, y2 = map(int, box.xyxy[0])   # box corners, pixel coords
#
#~ model.train(data=..., epochs=..., imgsz=..., batch=...)
#   Fine-tunes a pretrained model on your own dataset.
#   data   : path to a data.yaml describing train/val image folders + class names
#            (Roboflow's "YOLOv11" export format works directly)
#   epochs : how many full passes over the training data
#   imgsz  : training resolution (bigger = better accuracy, slower/more VRAM)
#   batch  : how many images per training step (limited by available VRAM/RAM)
#   Training writes runs/detect/train/weights/best.pt when finished.
#
# Example -- inference on a webcam:
#   model = YOLO('best.pt')
#   cap = cv.VideoCapture(0)
#   while cap.isOpened():
#       ret, frame = cap.read()
#       results = model(frame, conf=0.5, imgsz=320, verbose=False)
#       for box in results[0].boxes:
#           x1, y1, x2, y2 = map(int, box.xyxy[0])
#           cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
#       cv.imshow('YOLO', frame)
#       if cv.waitKey(1) & 0xFF == ord('q'):
#           break
#
# Example -- fine-tuning on a custom dataset:
#   model = YOLO('yolo11n.pt')
#   model.train(data='vehicles-openimages/data.yaml', epochs=30, imgsz=640, batch=4)
#
# See runnable version: 07_YOLO_Object_Detection/train_vehicle_detector.py
# (training only -- this repo doesn't currently include a standalone webcam
# inference script; the pattern above is exactly what one would look like).
