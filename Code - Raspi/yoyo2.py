import cv2
from ultralytics import YOLO


def main():
    # 1. Path to your custom-trained weights
    weights_path = r"C:\Users\Jakanat\Desktop\py.m5\runs\detect\box_detection_yolov11-5\weights\best.pt"

    # 2. Load the trained model
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    # 3. Initialize webcam (0 is usually the default built-i8 camera)
    cap = cv2.VideoCapture(0)

    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Optional: Set camera resolution (e.g., 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting webcam stream. Press 'q' or 'ESC' to exit.")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLOv11 inference on the frame
        # conf=0.5 filters out detections below 50% confidence (adjust as needed)
        results = model.predict(source=frame, conf=0.5, stream=True)

        # Plot detections on the frame
        for r in results:
            annotated_frame = r.plot()

        # Display the annotated frame
        cv2.imshow("YOLOv11 Live Box Detection", annotated_frame)

        # Press 'q' or ESC (27) to stop the program
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()