import cv2 as cv
import numpy as np

COLOR_ORDER = [
    ("yellow",      22.5),
    ("steel_blue",  45.0),
    ("black",       67.5),
    ("orange",      90.0),
    ("sky_blue",   112.5),
    ("pink",       135.0),
    ("coral",      157.5),
    ("green",      180.0),
]

PATCH = 30  # half-size of the sampling box in pixels

def sample_center(cap):
    # flush a few stale frames from the buffer, then read a fresh one
    for _ in range(5):
        ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read frame from camera.")
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    patch = frame[cy - PATCH:cy + PATCH, cx - PATCH:cx + PATCH]
    b, g, r = np.mean(patch, axis=(0, 1))
    return int(r), int(g), int(b)

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera at index 0.")
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    print("Color calibration. For each color, hold that paper square")
    print("centered in front of the camera, filling the middle of the frame,")
    print("then press Enter here in the terminal.\n")

    results = []
    for name, angle in COLOR_ORDER:
        input(f"Hold up '{name}' square, then press Enter...")
        r, g, b = sample_center(cap)
        print(f"  -> sampled RGB = ({r}, {g}, {b})\n")
        results.append((name, (r, g, b), angle))

    cap.release()

    print("\nCOLOR_TARGETS = [")
    for name, (r, g, b), angle in results:
        print(f'    ("{name}", ({r}, {g}, {b}), {angle}),')
    print("]")

if __name__ == "__main__":
    main()
