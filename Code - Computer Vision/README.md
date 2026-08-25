# Classwork | Computer Vision (OpenCV, Haar, MediaPipe, YOLO)

M.5 classwork progressing from raw OpenCV fundamentals through classical
face detection/recognition, MediaPipe hand tracking, and YOLO object
detection. Originally ~50 loose scripts in one folder; reorganized here by
topic, one lesson per file, everything gets a one-line header comment.

---

## What This Is

A single comprehensive reference (`OpenCV_Summary.py`) covering the whole
course arc — basics → face detection/recognition → MediaPipe → YOLO — plus
the individual exercise scripts that reference sections walk through,
grouped into numbered topic folders.

---

## Requirements

| Requirement | Value |
| :--- | :--- |
| Python | 3.x |
| pip packages | `opencv-python` (or `opencv-contrib-python` — **required** for `05_Face_Recognition_LBPH`, since `cv.face.LBPHFaceRecognizer_create()` only ships in the contrib build), `numpy`, `matplotlib`, `mediapipe` (06), `ultralytics` (07), `Pillow` (05 training step), `scikit-learn` (a few older 02/03 scripts import `sklearn.preprocessing.scale` but never call it — harmless unused import, not a hard requirement) |
| Hardware | Webcam for any `*_webcam.py`/`*_live.py` file. Raspberry Pi + `RPi.GPIO` only for `06_Hand_Tracking_Mediapipe/hand_tracking_color_to_servo.py`. |

---

## Start Here

**[`OpenCV_Summary.py`](OpenCV_Summary.py)** — read this first. One file, 17
sections, basics through YOLO/MediaPipe, every function's arguments
explained inline with copy-paste examples. Sections 1–14 have matching
runnable demos in the numbered folders below; sections 15–17 (face
recognition, hand tracking, YOLO) are reference/pseudocode pointing at the
real runnable lessons, since those need a webcam/dataset that can't just be
imported and run top-to-bottom like the earlier sections.

---

## Folder-by-Folder

### `01_Basics_IO_Drawing_Transforms/`
| File | What it does |
| :--- | :--- |
| `basics_and_transforms.py` | Full procedural walkthrough — read/show image & video, rescale, draw shapes, crop/translate/rotate/flip, color spaces, edge+contour pipeline, channels, bitwise, Sobel/Laplacian, mouse events. Everything inline, nothing imported. |
| `reusable_cv_toolkit.py` | Same operations as a function library (`load`, `show`, `rescale`, `canny`, `find_contours`, …) — `from reusable_cv_toolkit import *` instead of retyping cv2 calls. Has a `cheatsheet()` function and a runnable self-test. |
| `reusable_cv_toolkit_demo.py` | 13 numbered demo blocks exercising every `reusable_cv_toolkit.py` function. |

### `02_Drawing_and_Mouse_Exercises/`
Shape drawing + `cv.setMouseCallback` exercises: `polygon_fill.py`, `mouse_draw_rectangle.py`, `mouse_draw_ellipse.py`, `arrow_and_text.py`, `mouse_freehand_line.py`, `mouse_draw_triangle.py` (+ `_colored` variant), `mouse_pixel_color_picker.py`, `webcam_live_overlay.py` (live clock + shapes on the webcam feed).

### `03_Color_and_Bitwise/`
`bitwise_shapes_demo.py` (AND/OR/XOR/NOT on two shapes), `color_range_mask_bgr.py` (BGR `inRange`), `binary_threshold.py`, `pixel_color_picker_bgr.py`, `channel_min_max_analysis.py` (per-channel min/max + darkest/brightest pixel), `hsv_green_mask_image.py`, `hsv_multi_color_mask_image.py` (+ `_webcam` live version) — white/yellow/green HSV masking combined with `bitwise_or`.

### `04_Face_Detection_Haar/`
Haar cascade face + eye detection: `face_and_eye_detection_combo.py`, `eye_detection_image.py`, `face_detection_image.py`, `face_detection_webcam.py`, `face_hide_webcam.py` (blanks the face, keeps the rest), `face_mask_show_only_webcam.py` (shows only the face, blanks the rest). Both `.xml` cascade files live here now, next to the scripts that load them.

### `05_Face_Recognition_LBPH/`
Three-step pipeline, run in order:
1. `step1_capture_training_faces.py` — captures ~100 face crops per user ID into `data/`
2. `step2_train_recognizer.py` — trains an LBPH recognizer on `data/`, saves `classifier.xml`
3. `step3_recognize_live.py` — loads `classifier.xml`, labels faces live on webcam by name + confidence

`data/` (captured face photos) is gitignored — biometric training images shouldn't be committed. `classifier.xml` (~48MB trained model) **is** currently tracked; see Known Issues below.

### `06_Hand_Tracking_Mediapipe/`
`hand_tracking_basic_solutions_api.py` (old `mp.solutions.hands` API), `hand_tracking_new_tasks_api.py` (current `HandLandmarker` Tasks API replacement), `hand_tracking_optimized_color_sample.py` (frame-skipping + fingertip color sampling), `hand_tracking_color_to_servo.py` (matches sampled color to a palette, drives a Raspberry Pi servo — needs `RPi.GPIO`, Pi-only). `hand_landmarker.task` is the model file the Tasks-API scripts load.

### `07_YOLO_Object_Detection/`
`train_vehicle_detector.py` — fine-tunes `yolo11n.pt` on a custom vehicle dataset. **The dataset folder isn't in this repo** (too large) — export it from Roboflow into a `vehicles-openimages/` folder next to this script before running. No standalone inference script currently exists here; `OpenCV_Summary.py` Section 17 shows what one would look like.

### `midterm/`
`exam1.py` (fully commented-out early draft), `exam2.py` (working submitted version — batch Haar detection over `faces/img_NNN.jpg`, hardcoded to the original lab PC's path, needs `base_dir` updated to run elsewhere), plus a results screenshot. `output/` (annotated result images) is gitignored.

### `assets/`
Shared test images/video used across the lesson folders via relative `../assets/...` paths: `img.jpg`, `test.png`, `test2.png`, `col.jpg`, `97941_0.jpg`, `IMG_7847.JPG`, `testing.JPG` (currently unused by any script), `Screenshot 2026-06-24 124535.png`, `Sequence 01.mp4`.

---

## Known Issues / Possible Problems

- **`classifier.xml` (~48MB) is a trained face-recognition model tracked in git**, and `step2_train_recognizer.py`'s `NAMES` dict maps numeric IDs to real classmates' names. Neither is a secret exactly, but both encode personal biometric data tied to real people — worth gitignoring `classifier.xml` too (like `data/` already is) before this repo goes anywhere public, and reconsidering whether real names belong in source vs. a local-only config.
- **`face_and_eye_detection_combo.py` originally pointed at `haarcascade_frontalface_default (1).xml`/`haarcascade_eye_tree_eyeglasses (1).xml` via `cv.data.haarcascades + <relative path>`** — mixing OpenCV's built-in cascade directory with a project-relative suffix that doesn't exist there, and the `(1)` files were never actually present in this repo. Fixed to load the real `.xml` files that ship in this folder.
- **`midterm/exam2.py` and the LBPH scripts' original paths were hardcoded to a specific machine** (`C:\Users\SPSM LAB 7\...`, one original script literally named `OOP.py`/`OOP-2.py` despite having nothing to do with object-oriented programming). Recognition-pipeline scripts now resolve paths via `os.path.dirname(__file__)` so they work from any machine; `exam2.py` is a graded submission and was left with its original `base_dir` plus a comment, rather than silently rewritten.
- **`07_YOLO_Object_Detection/train_vehicle_detector.py` has no dataset checked in.** The `vehicles-openimages/` folder it needs must be exported from Roboflow separately — training will fail with a missing-file error until that folder exists next to the script.
- **No standalone YOLO *inference* script exists** — only the training script. `OpenCV_Summary.py` Section 17 documents the inference pattern, but there's nothing to just run against a webcam yet.
