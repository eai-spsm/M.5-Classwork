from ultralytics import YOLO

WEIGHTS_PATH = "best.pt"
IMG_SIZE = 320

# NCNN gives the best CPU inference speed on Raspberry Pi ARM boards.
# ONNX is included as a portable fallback if ncnn export/runtime isn't available.
EXPORT_FORMATS = ["ncnn", "onnx"]


def main():
    model = YOLO(WEIGHTS_PATH)

    for fmt in EXPORT_FORMATS:
        print(f"\nExporting to {fmt}...")
        try:
            path = model.export(format=fmt, imgsz=IMG_SIZE, half=False, simplify=True)
            print(f"Saved: {path}")
        except Exception as e:
            print(f"Export to {fmt} failed: {e}")


if __name__ == "__main__":
    main()
