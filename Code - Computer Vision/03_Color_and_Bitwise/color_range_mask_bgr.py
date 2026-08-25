# Lesson: cv.inRange() on raw BGR values to mask out a green range (HSV masking is more robust -- see hsv_*_mask files).
import cv2 as cv
import numpy as np


# Create two white canvases (500x500 pixels, 3 channels, 8-bit)
img1 = np.full((500,500,3), 255, dtype=np.uint8)
img2 = np.full((500,500,3), 255, dtype=np.uint8)
img3 = cv.imread('../assets/test2.png') # Load your source image

# --- 2. COLOR FILTERING (HSV/BGR Thresholding) ---
# Define the range of BGR value green color to detect 
upper_green = np.array([100, 255, 130]) #dark green
lower_green = np.array([0, 50, 0]) # mint green

# Create a binary mask: white pixels = green detected, black pixels = everything else
mask = cv.inRange(img3, lower_green, upper_green)

# # Apply the mask: only keep pixels where the mask is white (the green parts)
# result = cv.bitwise_and(img3, img3, mask=mask)

# # --- 3. DRAWING SHAPES FOR LOGIC TESTING ---
# # Draw a blue-ish rectangle on img1 and a circle on img2
# cv.rectangle(img1, (70, 70), (430, 430), (245, 160, 118), -1)
# cv.circle(img2, (250, 250), 200, (245, 160, 118), -1)

# # --- 4. BITWISE OPERATIONS ---
# # Intersection: Only shows area where both shapes overlap
# bitwise_and = cv.bitwise_and(img1, img2)

# # Union: Shows the combined area of both shapes
# bitwise_or = cv.bitwise_or(img1, img2) 

# # Inversion: Flips the colors of img1 (Black -> white)
# # If a pixel is White (255), NOT makes it Black (0)
# # If a pixel is 200, NOT makes it 55 (255 - 200)
# bitwise_not = cv.bitwise_not(img1)

# # Difference: Shows only areas where the shapes DO NOT overlap
# bitwise_xor = cv.bitwise_xor(img1, img2)

# # --- 5. DISPLAY RESULTS ---
# cv.imshow('Rectangle', img1)
# cv.imshow('circle', img2)
# cv.imshow('AND', bitwise_and)
# cv.imshow('OR', bitwise_or)
# cv.imshow('NOT', bitwise_not)
# cv.imshow('XOR', bitwise_xor)
# cv.imshow('origin', img3)
cv.imshow('detect color mask', mask)
# cv.imshow('result', result)

cv.waitKey(0)
cv.destroyAllWindows()