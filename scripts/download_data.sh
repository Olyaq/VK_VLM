#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# This script warms up the local HF cache.
# NOTE: deepvk/LLaVA-Instruct-ru may be gated (requires accepting conditions / sharing contact info).
# Open the dataset page in a browser and accept conditions, then run `huggingface-cli login`.

python - <<'PY'
from datasets import load_dataset
datasets_to_try = [
    ("deepvk/GQA-ru", "train"),
    ("deepvk/GQA-ru", "test"),
    ("deepvk/MMBench-ru", "dev"),
    ("deepvk/MMBench-ru", "test"),
    ("deepvk/LLaVA-Instruct-ru", "train"),
]
for name, split in datasets_to_try:
    try:
        print(f"==> Loading {name} [{split}] ...")
        ds = load_dataset(name, split=split)
        print("   ok:", ds)
    except Exception as e:
        print(f"   skipped: {name} [{split}] -> {type(e).__name__}: {e}")
PY

echo "✅ Done (whatever could be downloaded is cached)."
