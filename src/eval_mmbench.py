from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, BitsAndBytesConfig

from .utils.io import coerce_image
from .utils.metrics import parse_choice_letter


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="deepvk/llava-saiga-8b")
    ap.add_argument("--model", type=str, default=None, help="Path to LoRA adapters dir (or full model dir).")
    ap.add_argument("--split", type=str, default="dev")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--quant", type=str, default="none", choices=["none","8bit","4bit"])
    return ap.parse_args()


def find_option_cols(cols: List[str]) -> List[str]:
    letters = ["A","B","C","D","E"]
    return [c for c in letters if c in cols]


def find_answer_col(cols: List[str]) -> Optional[str]:
    for c in ["answer", "label", "correct", "gt_answer"]:
        if c in cols:
            return c
    # sometimes it is 'answer' in uppercase
    for c in cols:
        if c.lower() in {"answer","label","correct"}:
            return c
    return None


def find_question_col(cols: List[str]) -> Optional[str]:
    for c in ["question", "query", "instruction", "text"]:
        if c in cols:
            return c
    return None


def find_image_col(cols: List[str]) -> Optional[str]:
    for c in ["image", "img", "image_path", "path", "image_url"]:
        if c in cols:
            return c
    return None


def build_prompt(question: str, options: Dict[str, str]) -> str:
    # A simple 1-pass prompt; MMBench also proposes circular evaluation in official repo.
    opt_lines = "\n".join([f"{k}. {v}" for k, v in options.items()])
    return f"<image>\n{question.strip()}\n\nВарианты:\n{opt_lines}\n\nОтветь только буквой (A/B/C/D/E)."


@torch.inference_mode()
def main():
    args = parse_args()
    ds = load_dataset("deepvk/MMBench-ru", split=args.split)

    cols = list(ds.column_names)
    q_col = find_question_col(cols)
    img_col = find_image_col(cols)
    ans_col = find_answer_col(cols)
    opt_cols = find_option_cols(cols)

    if q_col is None or img_col is None or ans_col is None or not opt_cols:
        raise ValueError(f"Can't infer columns for MMBench-ru split={args.split}. Columns={cols}")

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

    if args.model and args.model != args.base_model:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load adapters from {args.model}: {e}")

    n = min(args.max_samples, len(ds))
    correct = 0
    skipped = 0

    for ex in tqdm(ds.select(range(n)), total=n, desc="MMBench-ru"):
        question = str(ex[q_col])
        gold = str(ex[ans_col]).strip().upper()

        # Extract options
        options = {k: str(ex[k]) for k in opt_cols}
        # Some rows might miss images or have URLs. We only handle local/PIL images.
        try:
            img = coerce_image(ex[img_col])
        except Exception:
            skipped += 1
            continue

        user_text = build_prompt(question, options)
        messages = [{"role": "user", "content": user_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(images=[img], text=text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=4)

        pred_text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = parse_choice_letter(pred_text)

        if pred and gold:
            correct += int(pred == gold)

    denom = (n - skipped) if (n - skipped) > 0 else 1
    acc = correct / denom
    print(f"MMBench-ru split={args.split} samples={n} used={n-skipped} acc={acc:.4f} (skipped={skipped})")


if __name__ == "__main__":
    main()
