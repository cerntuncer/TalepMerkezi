#!/usr/bin/env bash
set -euo pipefail

# Ensure we are in the script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Python venv
if [ ! -d .venv ]; then
	python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip and install deps
pip install --upgrade pip wheel
pip install -r requirements.txt

# Export optional model dir if user has one
export SEQ_CLS_MODEL_DIR=${SEQ_CLS_MODEL_DIR:-"$SCRIPT_DIR/models/seqcls"}

# Run uvicorn on localhost:8000
exec uvicorn main:app --host 127.0.0.1 --port 8000 --reload