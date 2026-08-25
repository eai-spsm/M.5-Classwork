from ultralytics import YOLO

DATA_PATH = "Yolo_setup/box.v5i.yolov11/data.yaml"
EPOCHS = 50
IMG_SIZE = 320
BATCH = 4


def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",
        workers=2,
        name="box_detection_raspi",
    )

    print("\nTraining completed successfully!")
    print(f"Results and weights saved in: {results.save_dir}")


if __name__ == "__main__":
    main()
