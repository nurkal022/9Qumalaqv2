# Тоғызқұмалақ MCTS: Нейросеть для национальной игры

## О проекте

Разработка искусственного интеллекта для тоғызқұмалақ — древней казахской настольной игры из семейства манкала. Проект объединяет классический игровой движок (NNUE + alpha-beta) с современным подходом AlphaZero (нейросеть + Monte Carlo Tree Search).

**Цель:** Создать сильнейший ИИ для тоғызқұмалақ, способный играть на уровне чемпионов.

---

## Архитектура системы

### Нейросеть: TogyzNet (2.2M параметров)

```
Вход: [7 каналов × 9 позиций] — полное состояние доски

Каналы:
  0: Лунки текущего игрока (нормализовано /50)
  1: Лунки соперника
  2: Казан текущего игрока (захваченные камни /82)
  3: Казан соперника
  4: Тұздық текущего игрока (one-hot)
  5: Тұздық соперника
  6: Индикатор стороны (1.0 = белые, 0.0 = чёрные)

Тело: 10 Residual Blocks × 192 канала (1D свёртки)
  Каждый блок: Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → skip → ReLU

Выходы:
  Policy head → 9 вероятностей (какой ход сделать)
  Value head → оценка позиции [-1, +1]
```

### MCTS движок (Rust)

Высокопроизводительный движок на Rust с GPU-ускорением:

- **Batch-parallel MCTS**: 128 позиций оцениваются одним GPU вызовом
- **ONNX Runtime**: 0.9ms на батч из 128 позиций (RTX 5080)
- **Gumbel MCTS**: Sequential Halving для эффективного выбора ходов
- **Multi-threaded selfplay**: 20 параллельных рабочих потоков

### Производительность

| Режим | Скорость |
|-------|---------|
| Selfplay (Gumbel MCTS) | 17 игр/сек |
| Selfplay (Deep MCTS 200 sims) | 1.6 игр/сек |
| GPU inference (batch 128) | 0.9 мс |
| Serve mode (CPU, 1-ply) | 30-80 мс/ход |

---

## Данные и обучение

### Источники данных

| Источник | Объём | Назначение |
|----------|-------|-----------|
| PlayOK 1500+ | 500K позиций (4,800 игр) | Supervised pretrain |
| PlayOK 2000+ | 128K позиций (1,253 игры) | Expert data mixing |
| Engine selfplay (Gen7, depth 10) | 1.4M позиций (10,000 игр) | Distillation |
| MCTS selfplay | 5M+ позиций | Reinforcement learning |
| Мастер-игры (сервер) | Накопление | Обучение на реальных партиях |

### Supervised Pretraining

Модель обучалась на партиях сильных игроков PlayOK:

```
Данные: 4,800 игр, оба игрока ≥ 1500 Elo
Эпохи: 50, Batch: 2048, LR: 0.002 → cosine decay
Результат: 68.7% точность предсказания хода (val)
```

### Distillation от Gen7 Engine

Передача знаний от NNUE engine к нейросети:

```
1. Engine играет 10,000 партий (depth 10, NNUE eval)
2. Записывается: позиция + лучший ход engine + оценка
3. Нейросеть учится предсказывать ходы и оценки engine
4. Policy target: лучший ход engine (one-hot)
5. Value target: λ × engine_eval + (1-λ) × game_result
```

### MCTS Self-Play Training

AlphaZero-стиль обучение через самоигру:

```
Цикл обучения:
1. SELFPLAY: 200 игр модель-vs-модель (Rust, GPU)
   → Policy target: improved policy из MCTS поиска
   → Value target: результат партии (score-proportional)
2. TRAINING: обновление весов (PyTorch, GPU)
3. EXPORT: конвертация в ONNX для Rust
4. EVAL: оценка силы vs engine
5. Повторить
```

### League Training

Гибридный подход — обучение против разных оппонентов:

```
Каждая итерация:
├── 60% игр: selfplay (модель vs модель)
├── 40% игр: vs Gen7 engine (через serve протокол)
└── + 15% expert data (PlayOK 2000+) в каждом batch
```

---

## Инфраструктура

### Rust MCTS Engine (`rust-mcts/`)

```
rust-mcts/
├── src/
│   ├── main.rs           # CLI: selfplay, eval, serve, league
│   ├── board.rs           # Правила тоғызқұмалақ (467 строк)
│   ├── encoding.rs        # Доска → тензор [7,9] для нейросети
│   ├── evaluator.rs       # Центральный GPU batch evaluator
│   ├── mcts.rs            # Deep MCTS с PUCT selection
│   ├── gumbel.rs          # Gumbel MCTS (Sequential Halving)
│   ├── self_play.rs       # Генерация тренировочных данных
│   ├── league.rs          # Игра vs engine для обучения
│   ├── eval_vs_engine.rs  # Оценка силы: модель vs engine
│   └── replay_buffer.rs   # Бинарный формат данных (63 байта/запись)
├── scripts/
│   ├── train_distillation.py  # Distillation от engine
│   ├── train_hybrid.py        # Гибридное обучение
│   ├── train_loop.py          # Полный цикл обучения
│   ├── export_onnx.py         # PyTorch → ONNX конвертация
│   └── collect_master_games.py # Сбор партий с сервера
└── Cargo.toml             # ort 2.0, crossbeam-channel, clap
```

### Режимы работы Rust binary

| Режим | Команда | Назначение |
|-------|---------|-----------|
| Selfplay | `rust-mcts --games 200` | Генерация тренировочных данных |
| Eval | `rust-mcts --eval --games 10` | Оценка силы vs engine |
| Serve | `rust-mcts --serve --model m.onnx` | Веб-сервер для онлайн-игры |
| League | `rust-mcts --league --engine-games 40` | Обучение vs engine |

### Веб-деплой

```
Сервер: 85.239.36.121:8080
├── Flask web server (Python)
├── MCTS engine (Rust binary, serve mode)
├── ONNX модель (8.9 MB)
├── Opening book (872 позиции)
└── Логирование мастер-партий
```

---

## Технические достижения

### Batch-Parallel MCTS

Вместо 800 последовательных GPU вызовов (как в наивном MCTS), наша реализация собирает 128 leaf-позиций за один обход дерева используя virtual loss, и оценивает их **одним GPU вызовом**. Это даёт ~100x ускорение по сравнению с последовательным подходом.

### Playout Cap Randomization (KataGo)

25% ходов: полный поиск (200 sims) → improved policy для обучения
75% ходов: быстрый ход (raw policy) → только value target

Это увеличивает throughput в ~4x без потери качества policy targets.

### Score-Proportional Values

Вместо бинарных +1/-1 (win/loss), используем пропорциональную оценку на основе разницы камней в казанах. Это даёт более богатый градиент для обучения value head.

### Gumbel MCTS (Sequential Halving)

Вместо традиционного UCB/PUCT selection, используем Gumbel noise + Sequential Halving. Гарантирует policy improvement даже с 16 симуляциями (вместо 800+).

### Color-Paired Evaluation

Каждый тест = пара игр из одной стартовой позиции с разменой цветов. Это устраняет bias от преимущества первого хода и даёт честную оценку силы.

---

## Результаты обучения

### Прогресс Policy Loss

```
Supervised pretrain:          1.68 (68.7% accuracy)
v3 после 500 итераций:       1.09 (значительное улучшение)
v5 Gumbel MCTS 823 итерации: 0.72 (модель точно предсказывает свои ходы)
v7 League training:           0.60 (рекордно низкий)
```

### Value Head

```
Supervised:    v_loss = 0.52
v4 (score-proportional): v_loss = 0.12 (4x лучше)
Distillation от engine: v_loss = 0.059 (9x лучше чем supervised)
```

### Ключевые метрики

| Версия | Подход | p_loss | Eval vs Gen7 |
|--------|--------|--------|-------------|
| Pretrain | Supervised PlayOK 1500+ | 1.68 | ~50% |
| v3 | Deep MCTS + playout cap + expert | 1.09 | 50-70% |
| v5 | Gumbel MCTS selfplay | 0.72 | 50% |
| v6 | League (40% vs engine) | 0.60 | 50%+ |
| Distilled | Engine depth 10 distillation | 1.28 | Baseline |

---

## Масштаб проекта

### Код

| Компонент | Язык | Строки |
|-----------|------|--------|
| Rust MCTS engine | Rust | ~2,500 |
| Training scripts | Python | ~2,000 |
| NNUE engine (Gen7) | Rust | ~5,000 |
| Web server | Python/HTML | ~700 |
| **Всего** | | **~10,200** |

### Вычисления

| Ресурс | Использовано |
|--------|-------------|
| GPU часы (RTX 5080) | ~200+ часов |
| Итерации обучения | 3,000+ |
| Партий сгенерировано | 500,000+ |
| Позиций обработано | 10M+ |
| Данные (PlayOK) | 360,000 партий |

### Железо

| Компонент | Спецификация |
|-----------|-------------|
| GPU | NVIDIA RTX 5080 Laptop (16 GB VRAM) |
| CPU | 24 ядра |
| RAM | 30 GB |
| Inference | 0.9 мс на batch 128 (GPU) |

---

## Технологический стек

- **Rust** — игровая логика, MCTS поиск, selfplay, serve mode
- **Python/PyTorch** — обучение нейросети, distillation
- **ONNX Runtime** — кросс-платформенный inference (GPU/CPU)
- **CUDA** — GPU ускорение inference
- **Flask** — веб-сервер для онлайн-игры
- **Git/GitHub** — контроль версий

---

## Онлайн демо

**http://85.239.36.121:8080** — играйте против ИИ в браузере

- Opening book покрывает первые ~18 ходов
- Нейросеть (2.2M параметров) выбирает ходы в mid/end-game
- Логирование партий для будущего обучения
- Мастера тоғызқұмалақ тестируют систему

---

## Будущие направления

1. **Увеличение модели** — архитектуры до 5.9M параметров (large5m) подготовлены
2. **GPU сервер** — deep MCTS с 600+ симуляциями для уровня чемпионов
3. **Обучение на мастер-играх** — непрерывный сбор данных от сильных игроков
4. **Transformer архитектура** — attention mechanism для игрового дерева
5. **PlayOK интеграция** — автоматическая игра для рейтинга
