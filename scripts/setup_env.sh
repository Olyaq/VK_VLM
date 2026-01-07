#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment ready."
echo "Tip: if you need HF gated datasets/models, run: huggingface-cli login"
