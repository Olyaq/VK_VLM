---
language:
- ru
tags:
- vision-language
- multimodal
- vqa
- llava
license: mit
---

# Model Card: <MODEL_NAME>

## Summary
<MODEL_NAME> — русскоязычная VLM, донастроенная на открытых данных VK (deepvk) для улучшения качества на GQA-ru и MMBench-ru.

## Base model
- Base: `<BASE_MODEL_NAME>`
- Training method: LoRA/QLoRA
- Format: LLaVA-style (`<image>` + chat template)

## Training data (open VK datasets)
- `deepvk/LLaVA-Instruct-ru` — instruction tuning (может быть gated)
- `deepvk/GQA-ru` — (опционально) адаптация на train
- `deepvk/MMBench-ru` — оценка

## Evaluation
| Benchmark | Metric | Score |
|---|---|---:|
| GQA-ru | ExactMatch(one-word) | TODO |
| MMBench-ru | Accuracy (1-pass) | TODO |

## Limitations
- Возможны галлюцинации и ошибки в редких доменах
- Не использовать для высокорисковых решений без человека

## Example usage
См. `src/eval_gqa.py` и `src/eval_mmbench.py`.
