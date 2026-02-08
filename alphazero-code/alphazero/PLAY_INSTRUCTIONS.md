# Как играть с обученной моделью

## Быстрый старт

### 1. Демо (показать первые ходы)
```bash
python play.py --mode demo --simulations 200
```

### 2. AI vs AI (посмотреть как играет)
```bash
python play.py --mode ai-vs-ai --games 3 --simulations 200
```

### 3. Играть против AI
```bash
# Играть белыми
python play.py --mode human-vs-ai --human-white --simulations 400

# Играть чёрными
python play.py --mode human-vs-ai --simulations 400
```

## Параметры

- `--checkpoint` - путь к модели (по умолчанию: `checkpoints/model_iter50.pt`)
- `--simulations` - количество MCTS симуляций (больше = сильнее, но медленнее)
  - 100-200: быстро, но слабее
  - 400-800: баланс
  - 1600+: максимальная сила
- `--games` - количество игр для AI vs AI

## Примеры

```bash
# Быстрое демо
python play.py --mode demo

# Сильная модель (800 симуляций)
python play.py --mode human-vs-ai --simulations 800

# Другая модель
python play.py --checkpoint checkpoints/model_iter20.pt --mode demo
```

## Советы

- Начни с `--simulations 200` для быстрой игры
- Для максимальной силы используй `--simulations 800`
- Модель думает ~1-3 секунды на ход (зависит от simulations)

