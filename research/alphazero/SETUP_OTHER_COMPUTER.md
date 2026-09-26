# Настройка на другом компьютере

## 1. Клонировать репозиторий и переключиться на ветку

```bash
git clone <repository-url>
cd 9Qumalaq
git checkout alphazero-training
```

## 2. Создать Conda окружение

```bash
conda create -n togyz-alphazero python=3.11 -y
conda activate togyz-alphazero
```

## 3. Установить зависимости

```bash
cd alphazero
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy onnx onnxruntime onnxscript
```

Или из requirements.txt (если есть):
```bash
pip install -r requirements.txt
```

## 4. Проверить установку

```bash
python test_setup.py
```

## 5. Продолжить обучение

```bash
# Продолжить с последнего чекпоинта
python train_fast.py \
    --resume checkpoints/model_iter150.pt \
    --iterations 250 \
    --games 300 \
    --simulations 200 \
    --batch-games 32 \
    --batch-size 1024

# Или использовать скрипт
./continue_training.sh
```

## 6. Экспортировать модель для браузера

```bash
python export.py \
    --checkpoint checkpoints/model_iter150.pt \
    --output browser_model/model.onnx \
    --model-size medium
```

## 7. Тестировать модель

```bash
# Быстрый тест
python test_alphazero_vs_levels.py \
    --checkpoint checkpoints/model_iter150.pt \
    --games 10

# Играть против модели
python play.py \
    --checkpoint checkpoints/model_iter150.pt \
    --mode human
```

## Важные файлы

- `checkpoints/model_iter150.pt` - последний чекпоинт (45k игр)
- `train_fast.py` - оптимизированный тренинг
- `test_alphazero_vs_levels.py` - тестирование против других AI
- `export.py` - экспорт в ONNX для браузера

## Системные требования

- CUDA 12.1+ (для GPU)
- Python 3.11
- PyTorch с CUDA поддержкой
- ~20GB свободного места (для чекпоинтов и данных)

