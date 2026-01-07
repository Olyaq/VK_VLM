# Отчёт по проекту: VK VLM (GQA-ru + MMBench-ru)

## 1. Постановка задачи
Цель: донастроить VLM на открытых данных VK (deepvk) и максимизировать метрики на GQA-ru и MMBench-ru.

## 2. Данные (VK / deepvk) и их использование
См. `docs/DATA.md`.

## 3. Архитектура и базовая модель
- Base model: <BASE_MODEL_NAME>
- Метод: LoRA/QLoRA
- Формат: LLaVA-style (используем `<image>` + chat template)

## 4. Метод обучения
- Instruction tuning (основной этап)
- (опционально) доменная адаптация на GQA-ru train
- Техника: gradient checkpointing, mixed precision, (опционально) 4-bit

## 5. Эксперименты
- Бейзлайн без обучения
- Абляции: LoRA rank/alpha, lr, промпт-шаблоны, длина контекста

## 6. Оценка и результаты
Таблица результатов см. `results/metrics.md`.

## 7. Воспроизводимость
См. `docs/REPRODUCIBILITY.md`.
