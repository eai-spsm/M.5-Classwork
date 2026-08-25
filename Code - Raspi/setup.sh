#!/usr/bin/env bash
# One-shot setup: checks Python version, creates the venv if missing,
# installs pinned deps from requirements.txt, verifies imports.
set -e

cd "$(dirname "$0")"

REQUIRED_PY="3.11"
VENV_DIR="venv"

echo "== Checking Python version =="
if command -v pyenv >/dev/null 2>&1 && pyenv versions --bare | grep -q "^3.11.9$"; then
    PYBIN="$(pyenv root)/versions/3.11.9/bin/python3"
else
    PYBIN="$(command -v python3)"
fi

PY_VERSION="$("$PYBIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$PY_VERSION" != "$REQUIRED_PY" ]; then
    echo "WARNING: found Python $PY_VERSION, but mediapipe==0.10.9 is only confirmed"
    echo "         compatible with Python $REQUIRED_PY.x on this platform (aarch64)."
    echo "         Install it with: pyenv install 3.11.9 && pyenv local 3.11.9"
    echo "         Continuing anyway with $PYBIN ..."
fi
echo "Using: $PYBIN ($PY_VERSION)"

echo ""
echo "== Setting up venv =="
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "No venv found at $VENV_DIR, creating one..."
    "$PYBIN" -m venv "$VENV_DIR"
else
    echo "venv already exists at $VENV_DIR, reusing it."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ""
echo "== Installing dependencies from requirements.txt =="
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "== Verifying imports =="
python3 -c "
import cv2, mediapipe, numpy
print('cv2:', cv2.__version__)
print('mediapipe:', mediapipe.__version__)
print('numpy:', numpy.__version__)
try:
    import RPi.GPIO as GPIO
    print('RPi.GPIO:', GPIO.VERSION)
except ImportError:
    print('RPi.GPIO: not available (fine off-Pi, scripts fall back to simulation mode)')
"

echo ""
echo "Setup complete. Activate with:  source $VENV_DIR/bin/activate"
