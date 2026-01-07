#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# GQA-ru evaluation (ExactMatch, one-word)
python -m src.eval_gqa --model artifacts/adapters/run_01 --base_model deepvk/llava-saiga-8b --split test --max_samples 200

# MMBench-ru evaluation (1-pass accuracy)
python -m src.eval_mmbench --model artifacts/adapters/run_01 --base_model deepvk/llava-saiga-8b --split dev --max_samples 200
