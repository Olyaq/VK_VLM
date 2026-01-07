# DATA.md — Использование открытых данных VK (deepvk)

## Источники (VK / deepvk)
- deepvk/LLaVA-Instruct-ru — instruction-датасет для LLaVA-style обучения (может быть gated)
- deepvk/GQA-ru — VQA-бенчмарк (train/test)
- deepvk/MMBench-ru — мультимодальный бенчмарк (обычно dev/test)

Коллекция: https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc

## Как использовались
### 1) Instruction tuning
- train: deepvk/LLaVA-Instruct-ru (conversations: human/gpt + изображение)

### 2) Опциональная адаптация под GQA-ru
- train: deepvk/GQA-ru train
- строго без использования test сплита при обучении

### 3) Оценка
- GQA-ru: test (ExactMatch для однословного ответа)
- MMBench-ru: dev/test (accuracy по выбору варианта A/B/C/...)

## Защита от утечки
- не используем тестовые ответы в обучении
- все эксперименты фиксируются конфигами и логами
