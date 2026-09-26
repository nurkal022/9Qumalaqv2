# 10. Трек AlphaZero

Каталог: [`alphazero-code/alphazero/`](../alphazero-code/alphazero/) — ~6900 строк
Python.

Это исследовательская ветка проекта: попытка проверить, обойдёт ли AlphaZero-подход
(MCTS + большая нейросеть) классическую связку alpha-beta + NNUE.

**Итог эксперимента: не обошёл.** Все проверенные конфигурации проиграли
NNUE-движку со счётом 0:20. Ниже — что именно пробовали и почему не получилось;
разбор полезен, чтобы не повторять.

## Сеть: TogyzNet

Файл: [`alphazero/model.py`](../alphazero-code/alphazero/model.py)

ResNet-подобная архитектура на одномерных свёртках с двумя головами.

```
вход (7 × 9)
    ↓ Conv1d(7→C, k=3) + BatchNorm + ReLU
    ↓ N × ResidualBlock(C)
    ├─→ policy head: Conv1d(C→32, k=1) → FC(32×9 → 9) → log_softmax
    └─→ value head:  Conv1d(C→4, k=1) → FC(4×9 → 64) → FC(64 → 1) → tanh
```

Размеры ([`create_model`, model.py:288](../alphazero-code/alphazero/model.py#L288)):

| Размер | Блоков | Каналов | Параметров |
|--------|--------|---------|-----------|
| small | MLP | 256→256→128 | ~30 тыс. |
| **medium** | **10 ResBlock** | **128** | **1 003 542** |
| large | 20 ResBlock | 256 | ~5 млн |

Основные эксперименты шли на medium.

## Кодирование позиции

[`encode_state`, game.py:219](../alphazero-code/alphazero/game.py#L219)

7 каналов × 9 позиций = 63 значения:

| Канал | Содержимое |
|-------|-----------|
| 0 | лунки ходящего (нормализованные) |
| 1 | лунки соперника |
| 2 | қазан ходящего (константа по всем 9 позициям) |
| 3 | қазан соперника |
| 4 | тұздық ходящего (one-hot) |
| 5 | тұздық соперника (one-hot) |
| 6 | сторона хода (константа) |

Как и в NNUE, представление **относительно ходящего**. Свёрточная структура
здесь оправдана: соседние лунки связаны механикой раздачи, и ядро размера 3
захватывает локальный паттерн.

## MCTS

Файл: [`alphazero/mcts.py`](../alphazero-code/alphazero/mcts.py)

Стандартный PUCT из AlphaZero ([`mcts.py:52`](../alphazero-code/alphazero/mcts.py#L52)):

```python
ucb_score = child.value + c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
```

Конфигурация ([`MCTSConfig`, mcts.py:14](../alphazero-code/alphazero/mcts.py#L14)):

| Параметр | Значение |
|----------|----------|
| `c_puct` | 1.5 |
| `dirichlet_alpha` | 0.3 |
| `dirichlet_epsilon` | 0.25 |
| `temperature` | 1.0 |

Шум Дирихле добавляется в корне для разнообразия self-play
([`add_dirichlet_noise`](../alphazero-code/alphazero/mcts.py#L67)).

## Хронология экспериментов

### Этап 1. Supervised pretraining

[`supervised_pretrain.py`](../alphazero-code/alphazero/supervised_pretrain.py)

Обучение medium-модели на **3.35 млн экспертных позиций** с PlayOK.
Результат: **67.6% top-1 совпадения** с ходом эксперта. Разумная база.

### Этап 2. Проверка против NNUE-движка

[`test_vs_nnue.py`](../alphazero-code/alphazero/test_vs_nnue.py)

```
MCTS (200 симуляций, pretrained) vs NNUE-движок (1 с/ход):
0 - 20 (0% побед)
```

Проиграны все партии без исключения.

### Этап 3. Self-play уточнение

5 итераций стандартного AlphaZero self-play дали **катастрофическое забывание**:

- Policy loss **вырос** с 1.25 до 1.42
- Self-play данные разрушили знания, полученные при supervised-обучении
- Модель стала слабее, а не сильнее

### Этап 4. Дистилляция в NNUE

[`generate_nnue_data.py`](../alphazero-code/alphazero/generate_nnue_data.py)

Попытка перелить знания большой сети в NNUE: 139 363 позиции из 1000 MCTS
self-play партий, значение сети `[-1,1]` переведено в сантипешки через обратную
сигмоиду.

Результат: полученный Gen5 NNUE оказался **слабее** Gen4 (37.5% в 20 партиях).
139 тысяч позиций на фоне основного датасета в 2.9 млн — слишком мало и слишком
шумно.

### Этап 5. Gumbel AlphaZero

[`gumbel_az.py`](../alphazero-code/alphazero/gumbel_az.py) (722 строки)

Самая проработанная попытка. Решала обе выявленные проблемы:

**Gumbel MCTS** (Danihelka et al., 2022) вместо обычного — гарантирует улучшение
политики при 16–32 симуляциях вместо 800+:

1. Сэмплирование гумбелевского шума `g(a)` для каждого действия
2. Отбор top-k по `g(a) + log π(a)`
3. Sequential Halving: бюджет симуляций делится на `log₂(k)` фаз, после каждой
   отбрасывается худшая половина
4. Улучшенная политика: `π'(a) ∝ exp(logits(a) + σ(q̂(a)))`

**Supervised replay buffer** против забывания: каждый обучающий батч на 70%
состоит из self-play данных и на 30% из 500 тысяч экспертных позиций, которые
никогда не вытесняются из буфера.

Конфигурация: medium, 32 симуляции, 100 партий на итерацию, batch 512, lr 0.001,
50 итераций.

Результаты обучения (50 итераций, 5000 партий, 124 минуты):

| Итерация | Policy loss | Value loss | Побед против random |
|----------|-------------|-----------|---------------------|
| 1 | 0.891 | 0.237 | — |
| 10 | 0.852 | 0.239 | 93.3% |
| **25** | **0.786** | **0.186** | **100.0%** |
| 30 | 0.781 | 0.184 | 96.7% |
| 50 | 0.818 | 0.188 | 91.7% |

Забывания **не произошло** — supervised replay сработал как задумано. Policy
loss улучшился на 11%, оптимум на итерациях 25–31, дальше лёгкое переобучение.

Проверка лучшего чекпоинта:

```
Gumbel AZ (итерация 25, 200 симуляций) vs NNUE-движок (1 с/ход):
0 - 20 (0% побед)
```

Снова полное поражение. Улучшения Gumbel и replay-буфера решили проблемы
**стабильности обучения**, но не устранили фундаментальный разрыв.

## Почему MCTS проиграл

Из [`FULL_REPORT.md`](../FULL_REPORT.md):

1. **Глубина поиска.** Alpha-beta достигает глубины 15+ за секунду. MCTS с 200
   симуляциями — эффективно 3–4.
2. **Эффективность оценки.** NNUE — 18 тысяч параметров на целочисленной
   арифметике, миллионы позиций в секунду. Сеть — 1 млн параметров, float на GPU,
   тысячи позиций в секунду. Разрыв в три порядка.
3. **Характер игры.** Тоғызқұмалақ вознаграждает глубокий тактический расчёт
   больше, чем распознавание паттернов. Это не го.
4. **Ветвление.** Максимум 9 ходов — достаточно узко, чтобы alpha-beta с хорошей
   сортировкой был крайне эффективен. Именно то преимущество MCTS, ради которого
   его придумали (огромное ветвление), здесь отсутствует.

Пункт 4 — главный. AlphaZero создавался для го с ветвлением ~250, где alpha-beta
бессилен. При ветвлении 5–8 классический поиск раскрывает дерево на порядки
глубже за то же время.

## Что осталось в репозитории

Код полностью сохранён, обученных весов нет — `checkpoints/` и `*.pt` в
[`.gitignore`](../.gitignore).

| Файл | Строк | Назначение |
|------|-------|-----------|
| [`gumbel_az.py`](../alphazero-code/alphazero/gumbel_az.py) | 722 | Gumbel AZ + supervised replay |
| [`train_fast.py`](../alphazero-code/alphazero/train_fast.py) | 763 | батчевый MCTS, параллельный self-play |
| [`train.py`](../alphazero-code/alphazero/train.py) | 402 | исходный цикл обучения |
| [`model.py`](../alphazero-code/alphazero/model.py) | 339 | архитектуры сетей |
| [`game.py`](../alphazero-code/alphazero/game.py) | 348 | правила игры на Python |
| [`mcts.py`](../alphazero-code/alphazero/mcts.py) | — | обычный MCTS |
| [`supervised_pretrain.py`](../alphazero-code/alphazero/supervised_pretrain.py) | 335 | предобучение на экспертных данных |
| [`generate_nnue_data.py`](../alphazero-code/alphazero/generate_nnue_data.py) | 275 | дистилляция в NNUE |
| [`export.py`](../alphazero-code/alphazero/export.py) | — | экспорт в ONNX для браузера |

Плюс диагностика и тесты: [`diagnose_model.py`](../alphazero-code/alphazero/diagnose_model.py),
[`test_mcts_values.py`](../alphazero-code/alphazero/test_mcts_values.py),
[`test_alphazero_vs_levels.py`](../alphazero-code/alphazero/test_alphazero_vs_levels.py),
[`test_game_logic.py`](../alphazero-code/alphazero/test_game_logic.py) и другие.

Сопутствующая документация модуля:
[`README.md`](../alphazero-code/alphazero/README.md),
[`ANALYSIS.md`](../alphazero-code/alphazero/ANALYSIS.md),
[`DIAGNOSIS.md`](../alphazero-code/alphazero/DIAGNOSIS.md),
[`OPTIMIZATIONS.md`](../alphazero-code/alphazero/OPTIMIZATIONS.md).

## Воспроизведение

```bash
conda activate togyz-alphazero
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install numpy tqdm tensorboard onnx

cd alphazero-code/alphazero
python train.py --model-size medium --games 100 --simulations 800 --iterations 100
tensorboard --logdir logs/
```

Экспорт обученной модели в браузер:

```bash
python export.py checkpoints/model_final.pt --output ../browser_model
```

После этого уровень «AlphaZero» во втором фронтенде заработает — см.
[09-web.md](09-web.md#уровни-ai).
