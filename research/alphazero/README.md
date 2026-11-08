# AlphaZero Тоғызқұмалақ

Реализация AlphaZero для традиционной казахской игры Тоғызқұмалақ (9 кумалак).

## Архитектура

- **Neural Network**: ResNet-style сеть с policy и value heads
- **MCTS**: Monte Carlo Tree Search управляемый нейросетью
- **Self-Play**: Самообучение через игру с собой

## Требования

- Python 3.11+
- PyTorch 2.0+ с CUDA
- NVIDIA GPU (рекомендуется RTX 3080+ или выше)

## Установка

```bash
# Активировать окружение
source ~/miniconda3/etc/profile.d/conda.sh
conda activate togyz-alphazero

# Установить зависимости (если не установлены)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy tqdm tensorboard onnx
```

## Быстрый старт

### 1. Тест на работоспособность

```bash
cd alphazero
python game.py      # Тест игровой логики
python model.py     # Тест нейросети
python mcts.py      # Тест MCTS
python self_play.py # Тест self-play
```

### 2. Запуск обучения

```bash
# Базовое обучение (рекомендуется для начала)
python train.py --model-size medium --games 100 --simulations 800 --iterations 100

# Быстрое тестирование
python train.py --model-size small --games 20 --simulations 200 --iterations 10

# Максимальная сила (долго, ~24-48 часов)
python train.py --model-size large --games 200 --simulations 1600 --iterations 500
```

### 3. Экспорт для браузера

```bash
python export.py checkpoints/model_final.pt --output ../browser_model
```

## Параметры обучения

| Параметр | Описание | Рекомендуемое значение |
|----------|----------|----------------------|
| `--model-size` | Размер сети (small/medium/large) | medium |
| `--games` | Игр за итерацию | 100-200 |
| `--simulations` | MCTS симуляций на ход | 800-1600 |
| `--iterations` | Всего итераций обучения | 100-500 |
| `--batch-size` | Размер батча | 256-512 |
| `--lr` | Learning rate | 0.001 |

## Структура файлов

```
alphazero/
├── game.py        # Игровая логика
├── model.py       # Нейросеть (ResNet)
├── mcts.py        # MCTS с нейросетью
├── self_play.py   # Self-play генерация данных
├── train.py       # Основной цикл обучения
├── export.py      # Экспорт в ONNX
├── checkpoints/   # Сохранённые модели
└── logs/          # Логи обучения
```

## Ожидаемые результаты

| Итерации | Игр | Win rate vs Random | Время (RTX 5080) |
|----------|-----|-------------------|------------------|
| 10 | 1,000 | ~70% | ~30 мин |
| 50 | 5,000 | ~90% | ~3 часа |
| 100 | 10,000 | ~95% | ~6 часов |
| 500 | 50,000 | ~99% | ~30 часов |

## Мониторинг обучения

```bash
# TensorBoard (в отдельном терминале)
tensorboard --logdir logs/
```

## Примечания

1. **GPU Memory**: Medium модель использует ~2GB VRAM, Large ~4GB
2. **Первые итерации**: Модель учится базовым правилам
3. **После 50 итераций**: Начинает понимать стратегию
4. **После 200 итераций**: Уровень сильного любителя
5. **После 500+ итераций**: Потенциально чемпионский уровень

## Интеграция с браузером

После экспорта, скопируй файлы из `browser_model/` в папку с игрой и подключи:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script src="alphazero-inference.js"></script>
<script>
const nn = new AlphaZeroNN();
await nn.load('model.onnx', 'metadata.json');

// В коде ИИ:
const {policy, value} = await nn.predict(gameState);
const bestMove = nn.getBestMove(policy, validMoves);
</script>
```

