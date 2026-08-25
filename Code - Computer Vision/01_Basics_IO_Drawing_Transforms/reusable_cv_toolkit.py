# Lesson: reusable OpenCV function library -- one small function per command
# (load/show/resize/crop/rotate/color/drawing/vision-pipeline/bitwise), so
# later scripts can `from reusable_cv_toolkit import ...` instead of
# retyping the same cv2 calls. See reusable_cv_toolkit_demo.py for usage.
"""
OpenCV toolkit -- one function per command. Import what you need:

    from reusable_cv_toolkit import load, show, wait, destroy, canny, find_contours, SAMPLES

Notes: channel order is BGR. img.shape is (H, W, C); cv.resize/dsize want (W, H).
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# SAMPLE PATHS  (shared media lives in ../assets)
# ----------------------------------------------------------------------------
class SAMPLES:
    DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
    img = os.path.join(DIR, 'img.jpg')
    png = os.path.join(DIR, 'test.png')
    video = os.path.join(DIR, 'Sequence 01.mp4')
    WEBCAM = 0


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def load(path, flag=cv.IMREAD_COLOR):       # flag: 1 color | 0 gray | -1 alpha
    img = cv.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f'imread got nothing from: {path}')
    return img


def save(path, img):
    return cv.imwrite(path, img)


# ----------------------------------------------------------------------------
# DISPLAY
# ----------------------------------------------------------------------------
def show(img, title='image'):
    cv.imshow(title, img)
    return img


def wait(ms=0):                             # 0 = wait forever; returns key code
    return cv.waitKey(ms) & 0xFF


def destroy():
    cv.destroyAllWindows()


def plot(img, title='image'):               # show via matplotlib (BGR -> RGB)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB) if img.ndim == 3 else img,
               cmap=None if img.ndim == 3 else 'gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# ----------------------------------------------------------------------------
# MOUSE EVENTS
# ----------------------------------------------------------------------------
def set_mouse(window, callback, param=None):    # callback(event, x, y, flags, param)
    cv.setMouseCallback(window, callback, param)  # window must already exist (imshow first)


# ----------------------------------------------------------------------------
# VIDEO
# ----------------------------------------------------------------------------
def video(source=SAMPLES.WEBCAM):           # path for a file, 0 for webcam
    return cv.VideoCapture(source)


def set_res(capture, width, height):        # best on live webcams
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def play(source=SAMPLES.WEBCAM, process=None, title='Video', delay=20, quit_key='d'):
    capture = source if isinstance(source, cv.VideoCapture) else cv.VideoCapture(source)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        cv.imshow(title, process(frame) if process else frame)
        if cv.waitKey(delay) & 0xFF == ord(quit_key):
            break
    capture.release()
    cv.destroyAllWindows()


# ----------------------------------------------------------------------------
# RESIZE
# ----------------------------------------------------------------------------
def rescale(img, scale=0.5):                # by factor; keeps aspect ratio
    w, h = int(img.shape[1] * scale), int(img.shape[0] * scale)
    interp = cv.INTER_AREA if scale < 1 else cv.INTER_CUBIC
    return cv.resize(img, (w, h), interpolation=interp)


def resize(img, width, height, interp=cv.INTER_AREA):   # exact (W, H)
    return cv.resize(img, (width, height), interpolation=interp)


def scale_matrix(img, sx=0.5, sy=0.5):      # resize via affine matrix diagonal
    mat = np.float32([[sx, 0, 0], [0, sy, 0]])
    return cv.warpAffine(img, mat, (img.shape[1], img.shape[0]))


# ----------------------------------------------------------------------------
# TRANSFORMS
# ----------------------------------------------------------------------------
def crop(img, x1, y1, x2, y2):              # img[y1:y2, x1:x2]
    return img[y1:y2, x1:x2]


def translate(img, x, y):                   # +x right, -x left, +y down, -y up
    mat = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(img, mat, (img.shape[1], img.shape[0]))


def rotate(img, angle, point=None):         # CCW degrees, around center if None
    h, w = img.shape[:2]
    if point is None:
        point = (w // 2, h // 2)
    mat = cv.getRotationMatrix2D(point, angle, 1.0)
    return cv.warpAffine(img, mat, (w, h))


def rotate90(img, code=cv.ROTATE_90_CLOCKWISE):     # _COUNTERCLOCKWISE / _180
    return cv.rotate(img, code)


def flip(img, code=0):                      # 0 vertical | 1 horizontal | -1 both
    return cv.flip(img, code)


# ----------------------------------------------------------------------------
# COLOR SPACES
# ----------------------------------------------------------------------------
def gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def hsv(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)


def lab(img):
    return cv.cvtColor(img, cv.COLOR_BGR2LAB)


def rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def convert(img, code):                     # any cv.COLOR_* code
    return cv.cvtColor(img, code)


# ----------------------------------------------------------------------------
# COLOR CHANNELS  (split / merge)
# ----------------------------------------------------------------------------
def split(img):                             # -> (b, g, r), each 1-channel gray
    return cv.split(img)


def merge(channels):                        # merge([b, g, r]) -> color image
    return cv.merge(channels)


def channel(img, idx):                      # one channel in ITS color (0 B,1 G,2 R)
    chans = cv.split(img)
    zero = np.zeros(img.shape[:2], dtype='uint8')
    parts = [zero, zero, zero]
    parts[idx] = chans[idx]
    return cv.merge(parts)


# ----------------------------------------------------------------------------
# CANVAS + DRAWING  (drawing is in-place; returns the image)
# ----------------------------------------------------------------------------
def blank(shape=(500, 500, 3)):             # (H,W,3) color | (H,W) mask
    return np.zeros(shape, dtype='uint8')


def blank_like(img, color=True):
    return np.zeros(img.shape if color else img.shape[:2], dtype='uint8')


def rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=2):   # -1 = filled
    cv.rectangle(img, pt1, pt2, color, thickness)
    return img


def circle(img, center, radius=40, color=(0, 255, 0), thickness=-1):
    cv.circle(img, center, radius, color, thickness)
    return img


def line(img, pt1, pt2, color=(255, 255, 255), thickness=3):
    cv.line(img, pt1, pt2, color, thickness)
    return img


def text(img, txt, org=(10, 30), scale=1.0, color=(255, 255, 0),
         thickness=2, font=cv.FONT_HERSHEY_COMPLEX):
    cv.putText(img, txt, org, font, scale, color, thickness)
    return img


# ----------------------------------------------------------------------------
# VISION PIPELINE
# ----------------------------------------------------------------------------
def blur(img, ksize=5):                     # ksize must be odd; bigger = blurrier
    return cv.GaussianBlur(img, (ksize, ksize), cv.BORDER_DEFAULT)


def canny(img, t1=125, t2=175):             # lower = more (fainter) edges
    return cv.Canny(img, t1, t2)


def sobel_x(img, ksize=3):                  # vertical edges (gradient along x)
    return cv.convertScaleAbs(cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize))


def sobel_y(img, ksize=3):                  # horizontal edges (gradient along y)
    return cv.convertScaleAbs(cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize))


def sobel(img, ksize=3):                    # combined x + y
    return cv.bitwise_or(sobel_x(img, ksize), sobel_y(img, ksize))


def laplacian(img, ksize=1):               # edges in all directions
    return cv.convertScaleAbs(cv.Laplacian(img, cv.CV_64F, ksize=ksize))


def threshold(img, thresh=150, maxval=255, type=cv.THRESH_BINARY):   # gray in
    _, out = cv.threshold(img, thresh, maxval, type)
    return out


def threshold_otsu(img):                    # auto-picks the cutoff; gray in
    _, out = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return out


def adaptive(img, block=11, c=3):           # per-region; handles uneven light
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, block, c)


def dilate(img, ksize=3, iterations=1):     # thicken white edges
    return cv.dilate(img, (ksize, ksize), iterations=iterations)


def erode(img, ksize=3, iterations=1):      # thin white edges
    return cv.erode(img, (ksize, ksize), iterations=iterations)


def find_contours(img, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE):
    return cv.findContours(img, mode, method)   # -> (contours, hierarchy)


def draw_contours(canvas, contours, idx=-1, color=(0, 0, 255), thickness=2):
    cv.drawContours(canvas, contours, idx, color, thickness)   # idx -1 = all
    return canvas


# ----------------------------------------------------------------------------
# BITWISE  (combine images/masks of the SAME size)
# ----------------------------------------------------------------------------
def bitwise_and(a, b, mask=None):           # keeps where BOTH are white
    return cv.bitwise_and(a, b, mask=mask)


def bitwise_or(a, b, mask=None):            # keeps where EITHER is white
    return cv.bitwise_or(a, b, mask=mask)


def bitwise_xor(a, b, mask=None):           # keeps where they DIFFER
    return cv.bitwise_xor(a, b, mask=mask)


def bitwise_not(a, mask=None):              # inverts (white <-> black)
    return cv.bitwise_not(a, mask=mask)


# ----------------------------------------------------------------------------
# HELP  (print the syntax cheat sheet:  preset.cheatsheet())
# ----------------------------------------------------------------------------
def cheatsheet():
    """Print every function with its signature + a one-line note."""
    print(_CHEATSHEET)


_CHEATSHEET = """
=== preset.py cheat sheet ===========================================
  from preset import *          # or import preset as p

-- IO ---------------------------------------------------------------
  load(path, flag=1)            read image   (1 color | 0 gray | -1 alpha)
  save(path, img)              write image to disk

-- DISPLAY ----------------------------------------------------------
  show(img, title='image')     pop a window (chainable, returns img)
  wait(ms=0)                   0 = wait for key; returns the key code
  destroy()                   close all windows
  plot(img, title='image')     show via matplotlib (auto BGR->RGB)

-- MOUSE EVENTS -----------------------------------------------------
  set_mouse(window, callback, param=None)   callback(event,x,y,flags,param)

-- VIDEO ------------------------------------------------------------
  video(source=0)              open a capture (path / 0 webcam)
  set_res(capture, w, h)       set capture resolution (webcams)
  play(source, process=None, title='Video', delay=20, quit_key='d')
                               loop frames; process(frame)->frame to transform

-- RESIZE -----------------------------------------------------------
  rescale(img, scale=0.5)      by factor (keeps aspect ratio)
  resize(img, w, h, interp=cv.INTER_AREA)   to exact (W, H)
  scale_matrix(img, sx=0.5, sy=0.5)         resize via affine matrix

-- TRANSFORMS -------------------------------------------------------
  crop(img, x1, y1, x2, y2)    img[y1:y2, x1:x2]
  translate(img, x, y)         +x right, -x left, +y down, -y up
  rotate(img, angle, point=None)            CCW deg, center if None
  rotate90(img, code=cv.ROTATE_90_CLOCKWISE)
  flip(img, code=0)            0 vert | 1 horiz | -1 both

-- COLOR ------------------------------------------------------------
  gray(img)  hsv(img)  lab(img)  rgb(img)
  convert(img, code)           any cv.COLOR_* code

-- COLOR CHANNELS ---------------------------------------------------
  split(img)                   -> (b, g, r)
  merge(channels)              merge([b, g, r]) -> color
  channel(img, idx)            one channel in its color (0 B, 1 G, 2 R)

-- CANVAS + DRAWING  (in-place, returns img) ------------------------
  blank(shape=(500,500,3))     black canvas  ((H,W) = mask)
  blank_like(img, color=True)  black canvas matching img
  rectangle(img, pt1, pt2, color=(0,255,0), thickness=2)   -1 = filled
  circle(img, center, radius=40, color=(0,255,0), thickness=-1)
  line(img, pt1, pt2, color=(255,255,255), thickness=3)
  text(img, txt, org=(10,30), scale=1.0, color=(255,255,0), thickness=2)

-- VISION PIPELINE --------------------------------------------------
  blur(img, ksize=5)           odd ksize; bigger = blurrier
  canny(img, t1=125, t2=175)   lower = more edges
  sobel_x / sobel_y / sobel(img, ksize=3)   gradient edges (x / y / both)
  laplacian(img, ksize=1)      all-direction edges
  threshold(img, thresh=150, maxval=255, type=cv.THRESH_BINARY)
  threshold_otsu(img)          auto cutoff
  adaptive(img, block=11, c=3) per-region (uneven light)
  dilate(img, ksize=3, iterations=1)        thicken edges
  erode(img, ksize=3, iterations=1)         thin edges
  find_contours(img, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
                               -> (contours, hierarchy)
  draw_contours(canvas, contours, idx=-1, color=(0,0,255), thickness=2)

-- BITWISE  (same-size inputs) --------------------------------------
  bitwise_and(a, b, mask=None)   keeps where BOTH white
  bitwise_or(a, b, mask=None)    keeps where EITHER white
  bitwise_xor(a, b, mask=None)   keeps where they DIFFER
  bitwise_not(a, mask=None)      inverts black <-> white
=====================================================================
"""


# ----------------------------------------------------------------------------
# SELF-TEST  (runs only when executed directly)
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    cheatsheet()

    img = rescale(load(SAMPLES.img), 0.5)
    edges = canny(blur(gray(img)))
    contours, _ = find_contours(edges)
    print(f'{len(contours)} contour(s) found')

    canvas = blank_like(img)
    draw_contours(canvas, contours, color=(0, 255, 0))

    show(img, '1. original')
    show(edges, '2. canny')
    show(canvas, '3. contours')
    wait()
    destroy()
