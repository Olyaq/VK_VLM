# VK VLM Project: LoRA fine-tuning on GQA-ru

This repository contains a small project for fine-tuning a vision-language model on the open VK dataset **GQA-ru**.

## What was done
- Base model: `deepvk/llava-gemma-2b-lora`
- Fine-tuning method: LoRA
- Train data: `deepvk/GQA-ru` (train_balanced_instructions + train_balanced_images join)
- Evaluation data: `deepvk/GQA-ru` (testdev_balanced_instructions + testdev_balanced_images join)
- Final merged model: `artifacts/merged/final`
- Predictions: `artifacts/eval_testdev_final_300.jsonl`

## Results (sampled testdev)
- N=300, seed=42
- acc=0.4633
- yes/no acc=0.6496
- other acc=0.3443

## How to run
Train:
```bash
TOKENIZERS_PARALLELISM=false bash scripts/train.sh
