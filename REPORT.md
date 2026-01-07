# Отчёт по проекту: дообучение VLM на GQA-ru (LoRA)

## 1) Цель
Дообучить визуально-языковую модель на открытых данных VK (GQA-ru), чтобы улучшить ответы на вопросы по изображениям на русском языке и проверить качество на testdev части датасета.

## 2) Задачи
1. Подготовить пайплайн обучения LLaVA-модели на GQA-ru (инструкции + изображения).
2. Реализовать корректный коллатор для LLaVA (placeholder `<image>`, корректные `pixel_values`, без конфликтов с truncation).
3. Обучить LoRA-адаптеры на подмножестве train.
4. Сохранить артефакты: LoRA-адаптеры и merged модель.
5. Провести оценку на testdev и сохранить предикты в файл.

## 3) Данные (открытые данные VK)
Использован датасет HuggingFace: `deepvk/GQA-ru`.

### Обучение
- Instructions: `train_balanced_instructions`
- Images: `train_balanced_images`
- Выполнен join по `imageId`/`id`
- Использовано `max_train_samples=2000`

### Оценка
- Instructions: `testdev_balanced_instructions`
- Images: `testdev_balanced_images`
- Join по `imageId`/`id`
- Оценка на подвыборке N=300 (seed=42)

## 4) Модель и метод
- Базовая модель: `deepvk/llava-gemma-2b-lora`
- Метод дообучения: LoRA
  - r=16, alpha=32, dropout=0.05
  - target_modules: q_proj, k_proj, v_proj, o_proj

## 5) Параметры обучения
Конфигурация: `configs/train_lora.yaml`
- epochs: 1
- batch_size: 1
- grad_accum: 2
- lr: 2e-4, scheduler: cosine, warmup_ratio: 0.03
- max_seq_len: 512
- image_size: 336
- gradient_checkpointing: true
- eval_strategy: no (для экономии памяти на MPS)

## 6) Результаты
Модель: `artifacts/merged/final` (копия merged/run_02)

Метрика: exact match (нормализованный), testdev sample N=300, seed=42:
- accuracy = **0.4633** (139/300)
- yes/no accuracy = **0.6496**
- other accuracy = **0.3443**

Файл с предиктами:
- `artifacts/eval_testdev_final_300.jsonl`

## 7) Краткий анализ ошибок
Типичные ошибки:
- путаница объектов мебели (диван/стол и т.п.)
- частичная нормализация ответа (“за очки” vs “очки”)
- ошибки в цветах и пространственных отношениях
- иногда ошибки в yes/no (иногда инверсия ответа)

## 8) Как воспроизвести
1) Обучение:
```bash
TOKENIZERS_PARALLELISM=false bash scripts/train.sh
