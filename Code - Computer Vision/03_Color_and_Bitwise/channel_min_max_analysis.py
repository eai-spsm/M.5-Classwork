# Lesson: cv.split() per-channel min/max analysis + darkest/brightest pixel lookup.
import cv2 as cv
import numpy as np

# Load your image
img_path = "../assets/col.jpg"
img = cv.imread(img_path)

if img is None:
    print(f"Error: Could not open image '{img_path}'.")
    exit()

# Split image into Blue, Green, Red channel matrices
b_channel, g_channel, r_channel = cv.split(img)

print("=" * 40)
print("  BGR CHANNEL YIELD ANALYSIS")
print("=" * 40)

# 1. Per-Channel Min/Max (Yield)
channels = {"Blue": b_channel, "Green": g_channel, "Red": r_channel}

for name, ch in channels.items():
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(ch)
    print(f"\n[{name} Channel]")
    print(f"  • Lowest Yield  (Min): {int(min_val):3d} at pixel {min_loc}")
    print(f"  • Highest Yield (Max): {int(max_val):3d} at pixel {max_loc}")

# 2. Overall Pixel Brightness Analysis (Grayscale conversion)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
min_lum, max_lum, min_lum_loc, max_lum_loc = cv.minMaxLoc(gray)

darkest_bgr = img[min_lum_loc[1], min_lum_loc[0]]
brightest_bgr = img[max_lum_loc[1], max_lum_loc[0]]

print("\n" + "=" * 40)
print("  OVERALL EXTREME PIXELS")
print("=" * 40)
print(f"Darkest Pixel  at {min_lum_loc} -> BGR: {tuple(darkest_bgr.tolist())}")
print(
    f"Brightest Pixel at {max_lum_loc} -> BGR: {tuple(brightest_bgr.tolist())}"
)