# Продолжение обучения AlphaZero

## ✅ Проверено

Все компоненты проверены и работают правильно:
- ✅ Загрузка чекпоинта работает
- ✅ Модель загружается корректно
- ✅ Optimizer и scheduler восстанавливаются
- ✅ Training loop продолжается с правильной итерации

## Как продолжить обучение

### Вариант 1: Использовать скрипт (проще)

```bash
cd alphazero
./continue_training.sh
```

### Вариант 2: Вручную

```bash
cd alphazero
~/miniconda3/envs/togyz-alphazero/bin/python train_fast.py \
    --resume checkpoints/model_iter50.pt \
    --model-size medium \
    --games 100 \
    --simulations 200 \
    --iterations 100 \
    --batch-size 512
```

## Параметры

- `--resume` - путь к чекпоинту (обязательно для продолжения)
- `--iterations` - **общее** количество итераций (не дополнительных!)
  - Если сейчас 50 итераций, а хочешь до 100, укажи `--iterations 100`
  - Скрипт автоматически продолжит с итерации 51 до 100
- `--games` - количество игр на итерацию (100 = быстро, 200 = качественнее)
- `--simulations` - MCTS симуляции (200 = быстро, 400 = сильнее)

## Рекомендации

### Для быстрого улучшения (1-2 часа):
```bash
python train_fast.py --resume checkpoints/model_iter50.pt \
    --iterations 75 --games 100 --simulations 200
```
→ Доведёт до 75 итераций (25 новых)

### Для качественного улучшения (4-6 часов):
```bash
python train_fast.py --resume checkpoints/model_iter50.pt \
    --iterations 100 --games 200 --simulations 400
```
→ Доведёт до 100 итераций (50 новых), больше игр и симуляций

### Для максимальной силы (12+ часов):
```bash
python train_fast.py --resume checkpoints/model_iter50.pt \
    --iterations 150 --games 200 --simulations 400
```
→ Доведёт до 150 итераций (100 новых)

## Мониторинг

Во время обучения будет показываться:
- Win rate vs Random (каждые 10 итераций)
- Loss (policy + value)
- Скорость (игр/сек)

**Ожидаемые результаты:**
- Итерация 50: ~90-100% vs Random
- Итерация 75: ~95-100% vs Random, начинает побеждать Hard Minimax
- Итерация 100: ~100% vs Random, побеждает Hard, конкурирует с Expert
- Итерация 150+: Побеждает Expert, конкурирует с Master

## Сохранение

Чекпоинты сохраняются:
- Каждые 20 итераций: `checkpoints/model_iterX.pt`
- Всегда: `checkpoints/model_latest.pt`

## Прерывание

Можно безопасно прервать (Ctrl+C) - последний чекпоинт сохранится.
Продолжить можно с любого сохранённого чекпоинта.

