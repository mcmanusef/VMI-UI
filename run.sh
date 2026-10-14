#!/usr/bin/env bash
# Start the VMI UI on Linux with the venv made by install.sh.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "No .venv here yet -- run ./install.sh first."
    exit 1
fi
exec .venv/bin/python UI.py "$@"
