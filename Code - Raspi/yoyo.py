from ultralytics import YOLO


def main():
    # 1. Load a pre-trained YOLOv11 model (Nano version recommended for starting/testing)
    # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    model = YOLO("yolo11n.pt")

    # 2. Path to your data.yaml file
    data_path = r"C:\Users\Jakanat\Desktop\py.m5\box.v4i.yolov11\data.yaml"

    # 3. Train the model
    results = model.train(
        data=data_path,  # Path to dataset configuration file
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image resolution (pixels)
        batch=16,  # Batch size (reduce to 8 or 4 if you run out of GPU memory)
        device=0,  # GPU device ID (use 'cpu' if you don't have an NVIDIA GPU)
        workers=8,  # Data loader worker threads
        name="box_detection_yolov11",  # Folder name where runs will be saved
    )

    print("\nTraining completed successfully!")
    print(f"Results and weights saved in: {results.save_dir}")


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    main()