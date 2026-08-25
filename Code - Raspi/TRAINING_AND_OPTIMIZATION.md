# Training & Optimization Scripts

## Background

`yoyo.py` and `yoyo2.py` are PC-only scripts (hardcoded Windows paths,
`device=0` GPU training) — not meant to run on the Pi. `yoyo.py` is the
original training script, `yoyo2.py` is a PC-side webcam test of the
trained weights. Both stay as-is for reference; `train_raspi.py` and
`optimize_model.py` below are the Pi-appropriate equivalents.

## `train_raspi.py`

Trains a YOLOv11n model on the Pi itself using the dataset checked into
`Yolo_setup/box.v5i.yolov11/data.yaml` (12 classes: airpods max, box, boxing
glove, mouse, pillow, pinkpillow, power bank, rubik, scissors, sushi, water
bottle, wd-40).

Differences from `yoyo.py`:

| Setting | `yoyo.py` (PC) | `train_raspi.py` (Pi) |
|---|---|---|
| `device` | `0` (NVIDIA GPU) | `cpu` |
| `imgsz` | 640 | 320 (less RAM/compute) |
| `batch` | 16 | 4 (Pi has limited RAM) |
| `workers` | 8 | 2 (Pi CPU core count) |
| `epochs` | 100 | 50 (CPU training is slow — full 100 epochs on a Pi CPU is impractical) |
| `data` path | absolute Windows path | relative path into the repo's `Yolo_setup/` dataset |

Training a YOLO model from scratch on a Raspberry Pi CPU is realistically
meant for small experiments or fine-tuning, not full production training —
expect it to be an order of magnitude slower than a GPU. For real training
runs, prefer a PC/cloud GPU (`yoyo.py`) and just deploy the resulting
`best.pt` to the Pi for inference.

Run:

```bash
source venv/bin/activate
python3 train_raspi.py
```

## `optimize_model.py`

Exports the trained `best.pt` (PyTorch) to lighter formats for faster
inference on the Pi's ARM CPU, via `ultralytics`' built-in exporter (which
uses `torch` under the hood):

- **NCNN** — Tencent's inference framework, generally the fastest CPU
  backend on ARM boards like the Pi; primary target format here.
- **ONNX** — portable fallback format in case the NCNN runtime isn't
  installed/available.

Run:

```bash
source venv/bin/activate
pip install ncnn onnx onnxsim   # export dependencies, one-time
python3 optimize_model.py
```

Output files land next to `best.pt` (e.g. `best_ncnn_model/`, `best.onnx`).
Swap `test13.py`/`test15.py` over to `YOLO("best_ncnn_model")` (or the
`.onnx` path) instead of `best.pt` once you've confirmed the exported model's
detections still look correct — inference should be noticeably faster than
running the raw `.pt` weights directly on the Pi CPU.
