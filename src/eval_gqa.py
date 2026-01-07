from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig

from .data.prompts import gqa_one_word_prompt
from .utils.io import coerce_image
from .utils.metrics import exact_match_one_word


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="deepvk/llava-saiga-8b")
    ap.add_argument("--model", type=str, default=None, help="Path to LoRA adapters dir (or full model dir).")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--quant", type=str, default="none", choices=["none","8bit","4bit"])
    return ap.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    # Load dataset
    ds = load_dataset("deepvk/GQA-ru", split=args.split)

    cols = list(ds.column_names)
    img_col = pick_col(cols, ["image", "img"])
    q_col = pick_col(cols, ["question", "query", "instruction", "text"])
    a_col = pick_col(cols, ["answer", "label", "target", "gold"])

    if img_col is None or q_col is None or a_col is None:
        raise ValueError(f"Can't infer columns in GQA-ru split={args.split}. Columns={cols}")

    # Load model
    quant_config = None
    if args.quant in {"8bit","4bit"}:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=(args.quant=="8bit"),
            load_in_4bit=(args.quant=="4bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model_id = args.model or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    processor = AutoProcessor.from_pretrained(model_id)

    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model if args.model else model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )

    # If adapters are provided, try to load them with PEFT (optional dependency at runtime)
    if args.model and args.model != args.base_model:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load adapters from {args.model}: {e}")

    n = min(args.max_samples, len(ds))
    correct = 0

    for ex in tqdm(ds.select(range(n)), total=n, desc="GQA-ru"):
        img = coerce_image(ex[img_col])
        question = str(ex[q_col])
        gold = str(ex[a_col])

        user_text = gqa_one_word_prompt(question)
        messages = [{"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(images=[img], text=text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=8)

        pred = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        correct += exact_match_one_word(pred, gold)

    acc = correct / n if n else 0.0
    print(f"GQA-ru split={args.split} samples={n} ExactMatch(one-word)={acc:.4f}")


if __name__ == "__main__":
    main()
