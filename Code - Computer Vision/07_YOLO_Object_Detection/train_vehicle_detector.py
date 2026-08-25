# Lesson: fine-tune YOLO11n on a custom vehicle dataset (Roboflow "vehicles-openimages" export).
# NOTE: the dataset folder isn't checked into this repo (too large) -- download/export it
# from Roboflow into a "vehicles-openimages/" folder next to this script before running.
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "vehicles-openimages", "data.yaml")

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=4,
    )
