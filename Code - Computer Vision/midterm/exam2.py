# Midterm: batch Haar face-detect over faces/img_001..120.jpg, save annotated copies to output/.
# NOTE: base_dir below is the original lab-PC path this was submitted with -- point it at a
# local folder containing a faces/ subfolder of your own images to actually run this.
import cv2 as cv
import os

# Base paths
base_dir = r"C:\Users\SPSM LAB 7\Documents\024329\computer_vision-main"
xml_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
output_dir = os.path.join(base_dir, "output")

# Ensure output folder exists
os.makedirs(output_dir, exist_ok=True)

# Load classifier once outside the loop
face_cascade = cv.CascadeClassifier(xml_path)

if face_cascade.empty():
    print(f"Error: Could not load cascade classifier from {xml_path}")
    exit()

detected = 0

for i in range(1, 121):
    # :03d automatically formats numbers into 3 digits (001, 010, 100)
    img_path = os.path.join(base_dir, "faces", f"img_{i:03d}.jpg")
    img = cv.imread(img_path)

    # Check if image loaded correctly
    if img is None:
        print(f"Warning: Could not read image at {img_path}")
        continue

    # Fix: OpenCV uses BGR by default
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Standard detector parameters (scaleFactor=1.1, minNeighbors=5)
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5
    )

    if len(faces) > 0:
        for x, y, w, h in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out_path = os.path.join(output_dir, f"img_{i}.jpg")
        cv.imwrite(out_path, img)
    else:
        print(f"No faces detected in image {i}")

    detected += len(faces)

print(f"\nTotal faces detected: {detected}")