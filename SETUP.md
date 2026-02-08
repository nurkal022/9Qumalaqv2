# Тоғызқұмалақ AI — Setup & Training

## Быстрый старт (RTX 5000)

```bash
git clone https://github.com/nurkal022/9Qumalaqv2.git
cd 9Qumalaqv2
pip install torch numpy tqdm
```

## 1. Парсинг партий мастеров

```bash
python parse_games.py
```

Создаст `parsed_games/` с тренировочными данными:
- `training_all.npz` — 81K позиций (926 игр)
- `training_elo2300.npz` — 32K позиций (402 игры мастеров)

## 2. Тренировка AlphaZero

### Быстрый тест (5 минут)
```bash
cd alphazero-code/alphazero
python train_fast.py --model-size small --simulations 100 --iterations 3 --games 10
```

### Полная тренировка
```bash
cd alphazero-code/alphazero
python train_fast.py --model-size medium --simulations 800 --iterations 200
```

Параметры:
- `--model-size`: small / medium / large
- `--simulations`: кол-во MCTS симуляций на ход (800 рекомендуется)
- `--iterations`: кол-во итераций (каждая = 100 self-play игр + обучение)
- `--games`: игр за итерацию (по умолчанию 100)
- `--resume путь`: продолжить с чекпоинта

На RTX 5000: ~5-15 мин/итерация, полная тренировка ~1-2 дня.

### Мониторинг
- Каждые 10 итераций — оценка vs Random (цель: >95%)
- Чекпоинты в `checkpoints/` каждые 20 итераций
- Логи в `logs/training_history.json`

## 3. Тесты

```bash
cd alphazero-code/alphazero
python test_mcts_values.py
```

6 тестов проверяют корректность MCTS:
- Направление терминальных значений
- Консистентность знаков
- Предпочтение выигрышных ходов
- MCTSParallel
- TrueBatchMCTS дерево и выигрышные ходы

## Что было исправлено

1. **Баг инверсии значений в MCTS** (`mcts.py`): нетерминальные значения из нейросети не инвертировались → дерево предпочитало плохие ходы
2. **TrueBatchMCTS был одноуровневым** (`train_fast.py`): вместо дерева MCTS был 1-ply бандит → сеть не учила планировать
3. **Гиперпараметры**: 200→800 симуляций, temp threshold 15→30

## Структура проекта

```
9Qumalaqv2/
├── parse_games.py              # Парсер партий PlayOK
├── mcts.txt, mcts 2-4.txt      # Партии мастеров (ELO 1200-2520)
├── temirtau.txt                 # Партии игрока temirtau
├── parsed_games/                # Выход parse_games.py
│   └── valid_games.json         # 926 валидных игр
└── alphazero-code/
    └── alphazero/
        ├── game.py              # Движок игры
        ├── model.py             # Нейросеть (ResNet)
        ├── mcts.py              # MCTS (исправлен)
        ├── train_fast.py        # Тренировка с GPU батчингом (исправлен)
        ├── train.py             # Альтернативный тренер
        ├── self_play.py         # Self-play воркер
        ├── play.py              # Игра против модели
        └── test_mcts_values.py  # Тесты MCTS
```
