# Reproducibility

## Environment
- Python 3.10+
- CUDA (если есть GPU)
- Для QLoRA: bitsandbytes + совместимая видеокарта

## Install
```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

## (Optional) HF login
Если датасет или модель gated:
```bash
huggingface-cli login
```

## Download/cache datasets
```bash
bash scripts/download_data.sh
```

## Train
```bash
bash scripts/train.sh
```

## Evaluate
```bash
bash scripts/evaluate.sh
```

## Determinism
- seed задаётся в config
- конфиги — `configs/`
- логи — `results/logs/`
