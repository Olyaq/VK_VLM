# Results

Score formula: **Score = 0.5·GQA + 0.5·MMBench**

| Run | Base | Train data (VK open) | GQA-ru ExactMatch | MMBench-ru Acc | Avg Score | Artifacts / Links | Notes |
|---:|---|---|---:|---:|---:|---|---|
| run_01 | deepvk/llava-gemma-2b-lora | deepvk/GQA-ru (train_balanced_instructions, 2000) | 0.4600 (testdev@300) | — | — | [preds](../submission/eval_testdev_final_300.jsonl) | baseline comparison run |
| run_02 (final) | deepvk/llava-gemma-2b-lora | deepvk/GQA-ru (train_balanced_instructions, 2000) | **0.4633** (testdev@300) | **0.6067** (dev@300) | **0.5350** | [GQA log](../submission/gqa_testdev_300.txt), [GQA preds](../submission/gqa_testdev_pred_300.jsonl), [MMBench log](../submission/mmbench_dev_300.txt), [final metrics](../submission/final_metrics.txt), [config](../submission/train_lora.yaml) | merged model used locally: `artifacts/merged/final` |

