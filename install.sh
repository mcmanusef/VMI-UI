#!/usr/bin/env bash
# Set up the VMI UI on Linux: create .venv next to this script and install
# requirements.txt into it. Safe to re-run (updates the existing venv).
# Doesn't use sudo -- any system packages it finds missing are listed with
# the command to install them.
set -euo pipefail
cd "$(dirname "$0")"

MIN_MAJOR=3
MIN_MINOR=13

# Pick the newest suitable interpreter on PATH.
PYTHON=""
for candidate in python3.14 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)" 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Python >= $MIN_MAJOR.$MIN_MINOR is required (found: $(python3 --version 2>&1 || echo none))."
    echo "On Ubuntu: sudo apt install python3.13 python3.13-venv  (deadsnakes PPA on older releases)"
    exit 1
fi
echo "Using $PYTHON ($("$PYTHON" --version))"

if [ ! -x .venv/bin/python ]; then
    rm -rf .venv
    if ! "$PYTHON" -m venv .venv; then
        rm -rf .venv
        version="$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
        echo
        echo "Could not create the venv. On Debian/Ubuntu install the venv package first:"
        echo "    sudo apt install python${version}-venv"
        exit 1
    fi
fi

.venv/bin/python -m pip install --upgrade --disable-pip-version-check pip
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt

# Qt's xcb plugin needs X11 libraries the PyQt5 wheels don't ship.
if command -v dpkg >/dev/null 2>&1; then
    missing=""
    for pkg in libxcb-xinerama0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
        libxcb-keysyms1 libxcb-render-util0 libxcb-cursor0; do
        dpkg -s "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
    done
    if [ -n "$missing" ]; then
        echo
        echo "Qt needs these system packages to open a window:"
        echo "    sudo apt install$missing"
    fi
fi

# Serial access for the power supply.
if ! id -nG | grep -qw dialout; then
    echo
    echo "To use the power supply, add yourself to the dialout group (then log out and back in):"
    echo "    sudo usermod -aG dialout \$USER"
fi

echo
echo "Installed. Start the app with ./run.sh"
