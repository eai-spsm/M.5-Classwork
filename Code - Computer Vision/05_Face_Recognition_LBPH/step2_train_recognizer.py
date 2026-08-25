# Lesson step 2/3: train an LBPH face recognizer on data/*.jpg (from step1) -> classifier.xml. Run step3 next.
import numpy as np
from PIL import Image
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, 'classifier.xml')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# user id -> name, matches ids captured by fac3.py
NAMES = {
    1: "zihan",
    2: "thames",
    3: "lew",
    4: "Jun",
    5: 'Nat',
    6:"Yimwan"
}

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    faces = []
    ids = []

    face_cascade = cv2.CascadeClassifier(XML_PATH)

    for image_path in path:
        img = Image.open(image_path).convert("L")
        imageNp = np.array(img, 'uint8')

        try:
            id = int(os.path.split(image_path)[1].split(".")[1])
        except (IndexError, ValueError):
            print(f"Skipping {image_path}: Filename format must be 'name.id.extension'")
            continue

        detected_faces = face_cascade.detectMultiScale(imageNp)
        for (x, y, w, h) in detected_faces:
            faces.append(imageNp[y:y+h, x:x+w])
            ids.append(id)

    if not faces:
        print("Error: No faces detected in training data.")
        return

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write(MODEL_PATH)
    print(f"Training Completed! Model saved as {MODEL_PATH}")
    print(f"Trained on {len(faces)} faces, ids: {sorted(set(ids.tolist()))} -> {NAMES}")

if __name__ == "__main__":
    train_classifier(DATA_DIR)
