from __future__ import annotations

import argparse
import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import yaml
from datasets import load_dataset, Image as HFImage
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from .data.preprocess import extract_llava_conversation
from .utils.io import coerce_image
from .utils.seed import set_seed


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    labels: torch.Tensor


class LlavaSFTCollator:
    def __init__(self, processor, tokenizer, max_seq_len: int = 2048):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_texts(self, user_text: str, assistant_text: str) -> Tuple[str, str]:
        prompt_msgs = [{"role": "user", "content": user_text}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        full_msgs = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        full_text = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )
        return prompt_text, full_text

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Robust image loading
        images: List[Any] = []
        kept: List[Dict[str, Any]] = []
        for f in features:
            try:
                images.append(coerce_image(f["image"]))
                kept.append(f)
            except FileNotFoundError:
                continue

        if not kept:
            raise ValueError("All images in the batch were missing on disk.")

        features = kept

        full_texts: List[str] = []
        prompt_lens: List[int] = []
        for f in features:
            prompt_text, full_text = self._build_texts(f["user_text"], f["assistant_text"])
            full_texts.append(full_text)
            p_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_lens.append(len(p_ids))

        # IMPORTANT for LLaVA: avoid processor text truncation (can break image token count).
        text_batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        # Process images only
        if hasattr(self.processor, "image_processor") and self.processor.image_processor is not None:
            image_batch = self.processor.image_processor(images=images, return_tensors="pt")
        else:
            image_batch = self.processor(images=images, return_tensors="pt")

        batch = dict(text_batch)
        if "pixel_values" in image_batch:
            batch["pixel_values"] = image_batch["pixel_values"]
        else:
            batch.update(image_batch)

        # Normalize pixel_values shape: must be 4D (B, C, H, W). Some processors may return 3D for batch_size=1.
        pv = batch.get("pixel_values")
        if pv is not None:
            import torch
            if isinstance(pv, list):
                pv = torch.stack(pv, dim=0)
            if hasattr(pv, "dim") and pv.dim() == 3:
                pv = pv.unsqueeze(0)
            batch["pixel_values"] = pv

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        for i, p_len in enumerate(prompt_lens):
            labels[i, : min(p_len, labels.size(1))] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def _load_dataset_with_retries(name: str, config: str, split: str, retries: int = 5):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(name, config, split=split)
        except Exception as e:
            last_err = e
            msg = str(e)
            if any(code in msg for code in ["502", "503", "504", "Bad Gateway"]):
                print(f"HF transient error on attempt {attempt}/{retries}: {msg[:140]}")
                time.sleep(2 * attempt)
                continue
            raise
    raise RuntimeError(f"Failed to load dataset after retries: {last_err}")


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    dcfg = cfg.get("data", {})
    mcfg = cfg.get("model", {})
    tcfg = cfg.get("train", {})
    lcfg = cfg.get("lora", {})
    ocfg = cfg.get("output", {})

    base_model = mcfg["base_model_name"]
    max_seq_len = int(mcfg.get("max_seq_len", 2048))
    quant = mcfg.get("quantization", "none")
    prec = mcfg.get("precision", "bf16")

    # dtype for weights
    dtype = torch.float16 if prec == "fp16" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    processor = AutoProcessor.from_pretrained(base_model)

    # patch_size fix (some checkpoints have None)
    if getattr(processor, "patch_size", None) is None:
        try:
            ps = getattr(getattr(processor, "image_processor", None), "patch_size", None) or 14
            if hasattr(processor, "image_processor") and processor.image_processor is not None:
                setattr(processor.image_processor, "patch_size", ps)
            setattr(processor, "patch_size", ps)
            print(f"⚠️ processor.patch_size was None; set to {ps}")
        except Exception:
            pass

    quant_config = None
    if quant in {"8bit", "4bit"}:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=(quant == "8bit"),
            load_in_4bit=(quant == "4bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=None,
        quantization_config=quant_config,
    )

    # pull patch_size from model if available
    try:
        vc = getattr(model.config, "vision_config", None)
        ps2 = getattr(vc, "patch_size", None) if vc is not None else None
        if ps2 is not None:
            if hasattr(processor, "image_processor") and processor.image_processor is not None:
                setattr(processor.image_processor, "patch_size", ps2)
            setattr(processor, "patch_size", ps2)
            print(f"✅ Set processor.patch_size from model.vision_config: {ps2}")
    except Exception as e:
        print(f"⚠️ Could not set patch_size from model config: {e}")

    # Move to MPS if available
    if torch.backends.mps.is_available():
        model.to("mps")

    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # LoRA
    if lcfg.get("enabled", False):
        lora_cfg = LoraConfig(
            r=int(lcfg.get("r", 8)),
            lora_alpha=int(lcfg.get("alpha", 16)),
            lora_dropout=float(lcfg.get("dropout", 0.05)),
            target_modules=list(lcfg.get("target_modules", [])),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Dataset load
    ds_name = dcfg["train_dataset"]
    ds_split = dcfg.get("train_split", "train")
    ds_cfg = dcfg.get("dataset_config")

    ds = load_dataset(ds_name, ds_cfg, split=ds_split) if ds_cfg else load_dataset(ds_name, split=ds_split)

    # Limit before join
    max_samples = dcfg.get("max_train_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples > 0 and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
            print(f"Limited dataset to {max_samples} samples")

    # GQA join (instructions -> images), lazy image refs
    if ds_name == "deepvk/GQA-ru" and ds_cfg and "instructions" in ds_cfg:
        images_cfg = ds_cfg.replace("instructions", "images")
        print(f"Joining GQA-ru images from config: {images_cfg}")

        needed_ids = set(ds["imageId"]) if "imageId" in ds.column_names else set()
        img_ds = _load_dataset_with_retries(ds_name, images_cfg, ds_split, retries=5)
        if "image" in img_ds.column_names:
            img_ds = img_ds.cast_column("image", HFImage(decode=False))

        imageid_to_img = {}
        for ex in img_ds:
            image_id = ex.get("imageId") or ex.get("image_id") or ex.get("id")
            if needed_ids and image_id not in needed_ids:
                continue
            img_ref = ex.get("image")
            if image_id is not None and img_ref is not None:
                imageid_to_img[image_id] = img_ref
            if needed_ids and len(imageid_to_img) >= len(needed_ids):
                break

        def attach_image(ex):
            ex["image"] = imageid_to_img.get(ex.get("imageId") or ex.get("image_id"))
            return ex

        ds = ds.map(attach_image)
        ds = ds.filter(lambda ex: ex.get("image") is not None)
        print(f"After join: {len(ds)} examples")

    # Convert to SFT schema
    def to_sft(ex: Dict[str, Any]) -> Dict[str, Any]:
        if "conversations" in ex:
            user_text, assistant_text = extract_llava_conversation(ex)
            img = ex.get("image") or ex.get("img") or ex.get("image_path") or ex.get("path")
            return {"image": img, "user_text": user_text, "assistant_text": assistant_text}

        img = ex.get("image") or ex.get("img")
        q = ex.get("question")
        a = ex.get("answer")
        if img is None or q is None or a is None:
            raise ValueError(f"Unknown example schema keys={list(ex.keys())}")

        user_text = f"<image>\n{str(q).strip()}\nОтветь одним словом."
        assistant_text = str(a).strip()
        return {"image": img, "user_text": user_text, "assistant_text": assistant_text}

    ds = ds.map(to_sft, remove_columns=ds.column_names)

    # Filter missing local paths (only for str/dict path)
    def image_is_available(ex: Dict[str, Any]) -> bool:
        img = ex.get("image")
        if img is None:
            return False
        if isinstance(img, str):
            return os.path.exists(img)
        if isinstance(img, dict) and img.get("path"):
            return os.path.exists(img["path"])
        return True

    before_n = len(ds)
    ds = ds.filter(image_is_available)
    after_n = len(ds)
    print(f"Filtered missing images: {before_n} -> {after_n}")
    if after_n == 0:
        raise SystemExit("Dataset became empty after filtering images.")

    # Split
    val_ratio = float(dcfg.get("val_ratio", 0.02))
    if val_ratio > 0:
        split = ds.train_test_split(test_size=val_ratio, seed=seed)
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds, val_ds = ds, None

    collator = LlavaSFTCollator(processor=processor, tokenizer=tokenizer, max_seq_len=max_seq_len)

    out_adapters = os.path.join(ocfg.get("adapters_dir", "results/adapters"), run_name)
    out_logs = ocfg.get("logs_dir", "results/logs")
    os.makedirs(out_adapters, exist_ok=True)
    os.makedirs(out_logs, exist_ok=True)

    # TrainingArguments compatibility (eval_strategy vs evaluation_strategy)
    sig = inspect.signature(TrainingArguments.__init__)
    strategy_key = "evaluation_strategy" if "evaluation_strategy" in sig.parameters else "eval_strategy"

    ta_kwargs = dict(
        output_dir=out_logs,
        run_name=run_name,
        num_train_epochs=float(tcfg.get("epochs", 1)),
        per_device_train_batch_size=int(tcfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("grad_accum", 2)),
        learning_rate=float(tcfg.get("lr", 2e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(tcfg.get("scheduler", "cosine")),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 200)),
        eval_steps=int(tcfg.get("save_steps", 200)) if val_ds is not None else None,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        # mixed precision ONLY for CUDA; accelerate complains on MPS
        bf16=((prec == "bf16") and torch.cuda.is_available()),
        fp16=((prec == "fp16") and torch.cuda.is_available()),
    )
    ta_kwargs[strategy_key] = ("steps" if val_ds is not None else "no")

    training_args = TrainingArguments(**ta_kwargs)


    # ---- Force-disable evaluation on MPS/low-memory setups ----
    # Some configs/versions may still keep evaluation_strategy='steps' and trigger eval OOM on MPS.
    try:
        if str(getattr(training_args, "evaluation_strategy", "no")) != "no" or str(getattr(training_args, "eval_strategy", "no")) != "no":
            pass
    except Exception:
        pass

    # If config requests no-eval, enforce it hard
    want_no_eval = str(cfg.get("train", {}).get("eval_strategy", cfg.get("train", {}).get("evaluation_strategy", "no"))) == "no"
    if want_no_eval:
        training_args.evaluation_strategy = "no"
        try:
            training_args.eval_strategy = "no"
        except Exception:
            pass
        training_args.eval_steps = None
        training_args.do_eval = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None if str(training_args.eval_strategy) == 'no' else val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save adapters (or full model)
    if lcfg.get("enabled", False):
        model.save_pretrained(out_adapters)
        tokenizer.save_pretrained(out_adapters)
        processor.save_pretrained(out_adapters)
        print(f"✅ Saved LoRA adapters to: {out_adapters}")
    else:
        trainer.save_model(out_adapters)
        tokenizer.save_pretrained(out_adapters)
        processor.save_pretrained(out_adapters)
        print(f"✅ Saved model to: {out_adapters}")


if __name__ == "__main__":
    main()
