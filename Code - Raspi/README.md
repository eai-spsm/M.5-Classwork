# Raspberry Pi 4 GPIO / Vision Exercise Set

Weekly GPIO exercises (`w1.py`–`w7b.py`), one task per file, covering digital
output, PWM, buttons, ultrasonic sensing, motor driving, MediaPipe hand
tracking with a servo, and an I2C LCD — the last one paired with a YOLO
webcam detector. Targets a **Raspberry Pi 4**, BCM pin numbering throughout.

---

## Hardware

- Raspberry Pi 4 (BCM2835/2711 GPIO, `/dev/gpiochip0`)
- USB webcam (UVC), shows up as `/dev/video0` once plugged in
- Various: LEDs, push buttons, HC-SR04 ultrasonic sensor, DC motor + driver (L298N-style,
  IN1/IN2/ENA), SG90-class servo, 16x2 I2C LCD (PCF8574 backpack)

## Environment

| Component | Version |
|---|---|
| OS | Debian GNU/Linux 13 (trixie), Raspberry Pi reference build |
| Architecture | aarch64 (64-bit) |
| Python | 3.11.9 (managed via `pyenv`, pinned by `.python-version` in this folder) |
| pip | 24.0+ (run `pip install --upgrade pip` if older) |
| opencv-python | 5.0.0 |
| mediapipe | 0.10.9 |
| numpy | 2.4.6 |
| RPi.GPIO | 0.7.1 |

Compatibility note: `mediapipe==0.10.9` is confirmed to have a prebuilt wheel for
`cp311` + `manylinux_aarch64`, which is why Python 3.11 is the target version here
(pip will fall back to building from source, or fail, on other Python versions —
stick to 3.11.x unless you've separately confirmed a mediapipe wheel exists for it).

### Setup from a clean Pi — one command

```bash
./setup.sh
```

This checks your Python version (warns if it's not 3.11.x, since that's the
confirmed-compatible version for `mediapipe==0.10.9` on aarch64), creates
`venv/` if it doesn't exist yet, installs everything pinned in
[`requirements.txt`](requirements.txt) into it, and verifies all imports
(`cv2`, `mediapipe`, `numpy`, `RPi.GPIO`) at the end. Safe to re-run any time
— it reuses the existing venv instead of recreating it, and reinstalling
already-satisfied pinned versions is a no-op.

`requirements.txt` pins exact versions confirmed working on this Pi (Debian
13/trixie, aarch64, Python 3.11.9):

```
numpy==2.4.6
opencv-python==5.0.0.93
mediapipe==0.10.9
RPi.GPIO==0.7.1
```

Don't forget the hand landmark model — `w6.py` needs `hand_landmarker.task`
present in this folder (already included here). Source if you ever need to
re-download it:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

### Setup manually (what `setup.sh` does, step by step)

```bash
# 1. Confirm/install Python 3.11.9 via pyenv (skip if already set up)
pyenv install 3.11.9
pyenv local 3.11.9      # writes .python-version in this folder

# 2. Create + activate the venv
python3 -m venv venv
source venv/bin/activate

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install:

```bash
python3 -c "import cv2, mediapipe, numpy, RPi.GPIO; print('all good')"
```

### Using the venv day-to-day

```bash
source /home/prayut/Documents/venv/bin/activate
# prompt should now show (venv) at the start of the line
python3 w6.py
```

When you're done:

```bash
deactivate
```

Note: there's also an `env/` folder in this directory — that one was built against
the **system** Python 3.13.5 with `--system-site-packages` (a different, older
setup, not the pyenv 3.11.9 one this README targets). Don't mix the two;
use `venv/` for anything in this project.

## I2C LCD setup (PCF8574 16x2 backpack)

`w7a.py`/`w7b.py` drive a 16x2 character LCD over I2C (PCF8574 backpack,
default address `0x27`, some boards use `0x3f`). One-time setup on the Pi:

```bash
sudo raspi-config
# Interface Options -> I2C -> Enable
sudo reboot
```

After reboot, confirm the bus and device are visible:

```bash
ls /dev/i2c*                 # expect /dev/i2c-1
sudo i2cdetect -y 1          # expect the LCD's address (e.g. 27) in the grid
```

If nothing shows up in `i2cdetect`, see the "Common errors" table below.

Then install the LCD library into the venv:

```bash
source venv/bin/activate
pip install rpi_lcd
```

(`rpi_lcd` pulls in `smbus` as a dependency automatically — this is separate
from `smbus2`, which `RPLCD` uses. Both can coexist in the same venv.)

## SSH / remote access

Connect from your computer:

```bash
ssh <pi-username>@<pi-ip-or-hostname>.local
```

If running headless (no monitor attached) and you need a GUI window (`cv.imshow`,
used in `w6.py`/`w7b.py`), the desktop session still runs virtually on `:0` even
without a monitor plugged in. Two ways to see it:

1. **Raspberry Pi Connect** (already set up on this Pi) — go to
   https://connect.raspberrypi.com in a browser, sign in, select this Pi,
   click "Screen sharing". Then run scripts with `DISPLAY=:0` prefixed:
   ```bash
   DISPLAY=:0 python3 w6.py
   ```
2. **VNC** — `wayvnc` is installed (this Pi runs the `labwc` Wayland compositor,
   so `wayvnc`, not RealVNC's X11 server, is the correct one).

Without a display at all, GUI calls like `cv.imshow()` will throw
`qt.qpa.xcb: could not connect to display` — either attach a display (real or
virtual, per above) or strip the `imshow` calls for headless operation.

## Common errors seen & fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'cv2'` / `'mediapipe'` | Active Python env had no packages installed at all | `pip install opencv-python mediapipe==0.10.9 numpy` into the active pyenv env |
| `qt.qpa.xcb: could not connect to display` | No `DISPLAY` env var set (e.g. running from an SSH/VS Code Remote terminal) | Prefix the command with `DISPLAY=:0`, and ensure a screen-share (Pi Connect) or VNC session is open so the window is actually visible |
| `Could not open camera device at index 0` / `getStreamChannelGroup Camera index out of range` | No `/dev/video0` present — either no camera plugged in, or another process already has it open | Check `lsusb` / `ls /dev/video*`; if a previous script instance is still running, `fuser /dev/video0` shows the PID holding it, `kill <pid>` frees it |
| `No cameras available!` (`rpicam-hello --list-cameras`) | No CSI ribbon camera and no USB webcam detected | Plug in a USB webcam (`lsusb` should show a "Web Camera" / UVC device) or reseat the CSI ribbon cable |
| `ImportError: No module named 'RPi'` | `RPi.GPIO` not installed in the active env | `pip install RPi.GPIO` |
| Servo has no holding torque after script exits | `servo.stop()` + `GPIO.cleanup()` stops PWM pulses; the horn drifts passively with no signal driving it | Expected behavior for GPIO.PWM servos — remove the final `stop()`/`cleanup()` if you want it to hold the last position after the script ends |
| `TypeError: unsupported operand type(s) for &: 'NoneType' and 'int'` during interpreter shutdown, after "Done, GPIO cleaned up" | Harmless double-cleanup: the `PWM` object's `__del__` tries to stop the channel again after `GPIO.cleanup()` already tore it down | Cosmetic only, safe to ignore |
| `Aborted` / `Fatal Python error: Aborted (core dumped)` right after a Qt plugin error | Same root cause as the `qt.qpa.xcb` display error above — Qt fails to initialize a platform plugin and the whole process aborts instead of raising a normal Python exception | Fix the display issue first (`DISPLAY=:0` + an active screen-share/VNC session); this isn't a separate bug, just how Qt fails when it truly can't find a screen |
| `cv2` underlined red / "Import \"cv2\" could not be resolved" in VS Code, even though `python3 -c "import cv2"` works fine in the terminal | VS Code/Pylance is pointed at a different Python interpreter than the one you installed packages into (e.g. system Python, or a different venv than `Documents/venv`) | In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → pick the interpreter matching `Documents/venv/bin/python3` (or wherever you actually installed the packages) |
| `fatal error: Python.h: No such file or directory` during `pip install` | pip couldn't find a prebuilt wheel for your exact Python version/architecture and fell back to compiling the package from source, which needs Python's C headers | Usually means your Python version doesn't match an available wheel — double check you're on Python 3.11.x (matches the confirmed `mediapipe==0.10.9` aarch64 wheel); if you really need to build from source, install headers first: `sudo apt install python3-dev` |
| `error: externally-managed-environment` when running `pip install <name>` directly on the system Python (not inside a venv/pyenv env) | Debian 12+/trixie's system Python blocks `pip install` outside a venv by default (PEP 668), to stop you from breaking apt-managed packages | **Prefer fixing the root cause, not this flag**: activate `Documents/venv` (or `pyenv local 3.11.9`) first, then `pip install` normally — it won't hit this error inside an isolated env. Only if you deliberately want to install into the system Python anyway, add `--break-system-packages` (e.g. `pip install <name> --break-system-packages`) — this disables the safety check, so double-check you're not about to clobber a package apt/other tools depend on |
| `sudo i2cdetect -y 1` shows an empty grid, no address at all | I2C interface disabled, wrong bus number, or a wiring/power issue on the LCD backpack | Enable via `sudo raspi-config` -> Interface Options -> I2C -> Enable, then `sudo reboot`; confirm `/dev/i2c-1` exists (`ls /dev/i2c*`) and `lsmod \| grep i2c_` shows `i2c_dev`/`i2c_bcm2835`; if the bus is fine but still nothing shows, check SDA/SCL wiring and that the backpack has power |
| `ModuleNotFoundError: No module named 'ultralytics'` (or any other pinned package) even though it's installed | Running with the wrong Python — e.g. the bare `pyenv` shim `python3` instead of the project's `venv/bin/python3` (there can be multiple envs in this folder: `venv/`, `env/`, system pyenv) | `source venv/bin/activate` first (prompt shows `(venv)`), then re-run; check with `pip show <package>` inside the activated env if unsure |
| `ModuleNotFoundError: No module named 'rpi_lcd'` | Library not installed in the active venv | `source venv/bin/activate && pip install rpi_lcd` |
| `qt.qpa.xcb: could not connect to display` / `Aborted` right after `cv.imshow()` when running `w7b.py` over SSH | Same root cause as the general Qt/display entry above — SSH session has no `DISPLAY` set | `DISPLAY=:0 python3 w7b.py` (needs an active screen-share/VNC session — see SSH/remote access section); if you don't actually need to *see* the detection window, strip the `cv.imshow`/`cv.waitKey` calls and run headless instead |

All the GPIO scripts follow the same shutdown shape:

```python
try:
    ...main loop...
except KeyboardInterrupt:
    print("\nProgram interrupted/stopped by user.")
finally:
    GPIO.cleanup()   # always release GPIO pins, even on crash/Ctrl+C
```

`GPIO.cleanup()` in `finally` is the important part — it releases pin state so
the next script run doesn't hit "channel already in use" warnings.

---

## File-by-file summary

One task per file, flat top-level scripts (no unnecessary function wrapping),
BCM pin numbering, wiring documented in each file's own header comment.

| File | Task | Wiring |
|---|---|---|
| `w1.py` | Terminal ATM (login, balance, deposit, withdraw with banknote breakdown) | No hardware — pure Python |
| `w2a.py` | Digital output: single LED blink | LED -> 220ohm -> GPIO18 -> GND |
| `w2b.py` | Analog-style output: PWM brightness fade 0-255 | LED -> 220ohm -> GPIO13 (PWM) -> GND |
| `w3a.py` | Two buttons, one LED (button 1 = on, button 2 overrides off while held) | LED on GPIO18; buttons on GPIO23/24, internal pull-up |
| `w3b.py` | 5-LED chaser: STEP button advances one LED, AUTO button runs the full chase, both = reset | LEDs on GPIO5/6/13/19/26; buttons on GPIO23 (step) / GPIO24 (auto) |
| `w4a.py` | HC-SR04 ultrasonic — print distance only | Trig=GPIO23, Echo=GPIO24 |
| `w4b.py` | HC-SR04 ultrasonic -> 5-LED range pattern (blink/outside-in/alternate/sequential-off by distance band) | Same sensor pins as `w4a.py`; LEDs on GPIO5/6/13/19/26 |
| `w5a.py` | DC motor driver — forward/backward via two buttons | ENA=GPIO13 (PWM), IN1=GPIO18, IN2=GPIO12; buttons on GPIO19/26 |
| `w5b.py` | DC motor driver — forward/backward/stop + speed via terminal input | Same driver wiring as `w5a.py`, no buttons |
| `w6.py` | MediaPipe hand tracking -> servo pointer (index fingertip x-position sets servo angle) | Servo signal on GPIO18; needs `hand_landmarker.task` in this folder |
| `w7a.py` | 16x2 I2C LCD — static text + live counter | LCD over I2C: SDA=GPIO2 (pin 3), SCL=GPIO3 (pin 5), fixed hardware bus pins |
| `w7b.py` | YOLO webcam detection -> shows detected class names on the same LCD | Same LCD wiring as `w7a.py`, plus a USB webcam |

`w5a.py`/`w5b.py` drive both `IN1`/`IN2` correctly for direction (an earlier
draft of this motor code only ever toggled `IN1`, which silently broke
direction control — worth knowing if you see that pattern elsewhere).
`w7a.py`/`w7b.py` use `rpi_lcd` (`LCD().text(str, line)`), not `RPLCD` —
the two LCD libraries have different APIs and aren't interchangeable.

### `calibrate_colors.py`
Standalone helper for tuning a `COLOR_TARGETS` list against the real camera —
prompts you, one color at a time, to hold a paper square centered in front of
the camera and press Enter, samples a 60x60px patch, and prints ready-to-paste
RGB constants. Written for an earlier color-matching + servo build; `w6.py`
points the servo at the index fingertip's screen position directly and
doesn't do color matching, so this script isn't wired into the current
exercise set — kept here in case that approach comes back.

### `requirements.txt`
Pinned dependency versions confirmed working on this Pi — install with
`pip install -r requirements.txt` inside an activated venv, or just run
`./setup.sh` (below) which does that for you.

### `setup.sh`
One-shot environment setup: checks the active Python is 3.11.x, creates
`venv/` if missing, installs `requirements.txt` into it, and verifies every
import (`cv2`, `mediapipe`, `numpy`, `RPi.GPIO`) works before finishing. Safe
to re-run any time — reuses an existing venv rather than recreating it.
