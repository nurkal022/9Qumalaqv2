# 11. Данные и обучение

Полный цикл: сырые партии людей и self-play движка → бинарный датасет →
обучение NNUE в PyTorch → квантованные веса → матч против предыдущей версии.

```
┌─ gameNew2/*.txt ──> parse_games.py ──> parsed_games/valid_games.json ─┐
│  (партии PlayOK)                                                      │
│                                                                        ├─> обучающий
├─ datagen (self-play) ──> *_training_data.bin ────────────────────────┤   датасет
│                                                                        │
└─ synthetic endgame ───────────────────────────────────────────────────┘
                                        │
                                        ▼
                          train_custom_k.py (PyTorch)
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
              nnue_weights.json                 nnue_weights.bin
                (отладка)                    (квантование ×64 → i16)
                                                        │
                                                        ▼
                                          match-nnue: новые vs старые веса
```

---

# Источники данных

## Партии живых игроков

[`gameNew2/`](../gameNew2/) — текстовые выгрузки партий с PlayOK от семи игроков:
`argenby`, `avataar`, `darru`, `korgol`, `mugalim`, `nowkie`.

Некоторые файлы продублированы (`mugalim (1..5).txt` — пять байт-в-байт
одинаковых копий, `argenby.txt` и `argenby (1).txt`). Парсер это учитывает: в
[`parse_games.py:158-161`](../parse_games.py#L158-L161) есть отпечаток партии, по
которому дубликаты отбрасываются:

```python
def game_fingerprint(moves, white, black):
    key = f"{white}|{black}|{''.join(str(m) for m in moves[:20])}"
    return hashlib.md5(key.encode()).hexdigest()
```

Отпечаток берёт имена игроков и первые 20 ходов. Партии-дубликаты не попадут в
датасет по нескольку раз.

## Self-play движка

[`engine/src/datagen.rs`](../engine/src/datagen.rs) (395 строк)

```bash
cd engine
./target/release/togyzkumalaq-engine datagen 10000 8 4 local
#                                     партий глубина потоков префикс
```

Параметры генерации:

| Параметр | Значение |
|----------|----------|
| Потоки | настраиваемое число, каждый играет независимо |
| TT на поток | 4 МБ |
| Разнообразие дебютов | первые 8 ходов — случайные |
| Adjudication | оценка > ±8500 четыре хода подряд → досрочный конец |
| Лимит партии | 300 полуходов |
| Калибровка | нормализация оценок к средней абсолютной ≈ 500 |

Случайные первые ходы нужны, чтобы движок не играл одну и ту же партию тысячу
раз. Adjudication экономит время на заведомо решённых позициях.

## Синтетические эндшпили

[`gen_endgame_data.py`](../engine/gen_endgame_data.py) генерирует случайные
глубокие эндшпили и позиции у самой границы победы. Эндшпильных позиций в
естественном self-play мало, а именно там NNUE слабее всего.

## Сводка по объёмам

Из [`REPORT.md`](../REPORT.md):

| Источник | Позиций |
|----------|---------|
| Self-play HCE (V1) | 1.24 млн |
| Self-play HCE (Mac) | 2.46 млн |
| Combined HCE | 3.71 млн |
| V7 improved self-play (с NNUE и детектом повторений) | 3.37 млн |
| Партии мастеров (Elo ≥ 1800) | ~100 тыс. |
| Партии всех рейтингов | ~200 тыс. |
| Синтетические эндшпили | ~800 тыс. |

---

# Формат обучающих данных

**26 байт на позицию** ([`datagen.rs:6-15`](../engine/src/datagen.rs#L6-L15)):

| Смещение | Размер | Поле |
|----------|--------|------|
| 0–8 | 9 | `pits_white[9]` (u8) |
| 9–17 | 9 | `pits_black[9]` (u8) |
| 18 | 1 | `kazan_white` (u8) |
| 19 | 1 | `kazan_black` (u8) |
| 20 | 1 | `tuzdyk_white` (i8, −1 = нет) |
| 21 | 1 | `tuzdyk_black` (i8) |
| 22 | 1 | `side_to_move` (u8) |
| 23–24 | 2 | `eval` (i16 LE, с точки зрения ходящего) |
| 25 | 1 | `result` (0 = проигрыш белых, 1 = ничья, 2 = победа белых) |

Каждая запись хранит **и оценку поиска, и итог партии** — обучение использует
взвешенную комбинацию обоих сигналов.

Формат фиксированной длины без заголовка: число позиций = размер файла / 26,
читать можно с любого смещения, кратного 26. Удобно для `mmap` и разбиения на
батчи.

---

# Обучение NNUE

## Функция потерь

Из [`REPORT.md`, раздел 6.1](../REPORT.md), реализация —
[`train_custom_k.py:93`](../engine/train_custom_k.py#L93):

```python
def sigmoid_eval(x, k=K):
    return torch.sigmoid(x / (k / 4.0))

pred_wp = sigmoid_eval(prediction)
eval_wp = sigmoid_eval(evals)
target  = lam * eval_wp + (1 - lam) * game_result
loss    = mean((pred_wp - target)**2 * sample_weights)
```

Идея: и предсказание сети, и оценку поиска пропускают через сигмоиду, переводя в
шкалу «вероятность победы». Обучение идёт в этой шкале, а не в шкале оценок —
разница между +50 и +100 сантипешками важна, а между +3000 и +3050 нет, и
сигмоида это отражает автоматически.

| Параметр | Роль |
|----------|------|
| `λ = 0.75` | 75% сигнала — оценка движка, 25% — фактический результат |
| Adaptive λ | для позиций без оценки (`eval = 0`) — `λ = 0`, чистый результат |
| Sample weights | эндшпиль ≤30 камней — вес 2×, ≤15 камней — 3× |

## Параметр K — самое важное число

`K` задаёт крутизну сигмоиды. Исправление **K = 1050 → K = 400** дало
**+256 Elo** — крупнейший единичный прирост в истории проекта.

Механика: при слишком большом K сигмоида почти линейна в рабочем диапазоне, и
сеть тратит ёмкость на точное воспроизведение больших оценок вместо того, чтобы
различать близкие позиции. Правильный K концентрирует градиент там, где партия
действительно решается.

Скрипт принимает K аргументом ([`train_custom_k.py:88`](../engine/train_custom_k.py#L88)):

```bash
python3 train_custom_k.py 400 0.75 nnue_weights.bin
#                          K    λ   выход
```

## Гиперпараметры

| Параметр | Значение | Примечание |
|----------|----------|------------|
| Batch size | **4096** | критично: 8192 теряет **~258 Elo** |
| Learning rate | 0.001 | Adam |
| Weight decay | 1e-5 | |
| Scheduler | CosineAnnealingLR | LR → 0 к концу |
| Epochs | 100 | рабочий диапазон 75–150 |
| Lambda | 0.75 | |
| Validation | 10% данных, максимум 50 тыс. | hold-out для сохранения лучших весов |

Разница в 258 Elo между batch 4096 и 8192 при прочих равных — самый
контринтуитивный результат в проекте. Больший батч даёт меньше шагов оптимизации
и меньше градиентного шума, а шум здесь, по-видимому, работает как регуляризация.

## Экспорт весов

Два формата:

- [`nnue_weights.json`](../engine/nnue_weights.json) (424 КБ) — полная точность,
  для отладки и диффов
- [`nnue_weights.bin`](../engine/nnue_weights.bin) (37 510 байт) — квантование
  float → int16 умножением на 64; это то, что читает движок

Формат бинарника описан в [06-nnue.md](06-nnue.md#бинарный-формат).

---

# Инструменты в `engine/`

## Обучение

| Скрипт | Назначение |
|--------|-----------|
| [`train_custom_k.py`](../engine/train_custom_k.py) | **основной тренер** с настраиваемыми K и λ |
| [`train_nnue_v2.py`](../engine/train_nnue_v2.py) | исходный тренер (K = 1050, устарел) |
| [`train_endgame.py`](../engine/train_endgame.py) | с усиленным весом эндшпильных позиций |
| [`train_dropout.py`](../engine/train_dropout.py) | эксперимент с dropout |
| [`train_multiseed.py`](../engine/train_multiseed.py) | несколько сидов подряд (разброс огромен) |
| [`train_gen8.py`](../engine/train_gen8.py), [`train_gen8_lr.py`](../engine/train_gen8_lr.py) | поколение 8 |
| [`finetune_nnue.py`](../engine/finetune_nnue.py) | дообучение существующих весов |
| [`finetune_transfer.py`](../engine/finetune_transfer.py), [`transfer_58feat.py`](../engine/transfer_58feat.py) | перенос весов на 58-признаковую архитектуру |

## Подготовка данных

| Скрипт | Назначение |
|--------|-----------|
| [`prepare_v8_data.py`](../engine/prepare_v8_data.py), [`prepare_v9_data.py`](../engine/prepare_v9_data.py), [`prepare_v9b_data.py`](../engine/prepare_v9b_data.py) | объединение источников с дупликацией эндшпилей 2–3× |
| [`merge_data.py`](../engine/merge_data.py) | склейка бинарников |
| [`convert_human_games.py`](../engine/convert_human_games.py), [`convert_master_games.py`](../engine/convert_master_games.py) | партии людей → бинарный формат |
| [`extract_expert_positions.py`](../engine/extract_expert_positions.py) | выборка экспертных позиций |
| [`gen_endgame_data.py`](../engine/gen_endgame_data.py) | синтетические эндшпили |

## Оценка силы

| Скрипт | Назначение |
|--------|-----------|
| [`match_engines.py`](../engine/match_engines.py) | **основной A/B-стенд**: матч двух сборок/сетей по официальным правилам, парные дебюты, параллельные партии, Elo ± 95% ДИ и SPRT |
| [`match_search.py`](../engine/match_search.py) | сравнение настроек поиска |
| [`selfplay_loop.py`](../engine/selfplay_loop.py) | цикл «генерация → обучение → проверка» |
| [`pipeline.py`](../engine/pipeline.py) | полный автоматизированный пайплайн (551 строка) |

Встроенные в движок команды:

```bash
./togyzkumalaq-engine match 100 1000                        # 100 партий по 1 с
./togyzkumalaq-engine match-nnue old.bin new.bin 100 1000   # сравнение весов
```

### Как проверять изменения

50–100 партий дают доверительный интервал ±70–100 Elo — этого не хватает,
чтобы отличить улучшение от шума (отсюда «разброс по сидам» и противоречивые
выводы в [12-results.md](12-results.md)). Любое изменение поиска, оценки или
весов проверяйте через `match_engines.py`:

```bash
cd engine
cp target/release/togyzkumalaq-engine /tmp/engine_base    # эталон ДО изменения
# ... правки, cargo build --release ...
python3 match_engines.py --a target/release/togyzkumalaq-engine --b /tmp/engine_base \
    --games 2000 --time 100 --elo0 0 --elo1 10
# разные сети: каталоги со своими nnue_weights.bin
python3 match_engines.py --a ./target/release/togyzkumalaq-engine --dir-a netA/ \
    --b ./target/release/togyzkumalaq-engine --dir-b netB/
```

Каждая пара партий играется с одним и тем же случайным дебютом и сменой цвета;
SPRT останавливает матч, как только результат статистически ясен. Принимать
изменение — только при `H1 accepted`.

---

# Анализ партий

| Скрипт | Назначение |
|--------|-----------|
| [`parse_games.py`](../parse_games.py) | парсинг PlayOK, валидация, дедупликация, фильтр по Elo |
| [`analyze_games.py`](../analyze_games.py) | статистика: винрейты, частоты ходов |
| [`analyze_deep.py`](../analyze_deep.py) | углублённый анализ (28 КБ) |
| [`export_positions.py`](../export_positions.py) | экспорт позиций |
| [`validate_perft.py`](../validate_perft.py) | проверка правил через perft |

Именно [`analyze_games.py`](../analyze_games.py) дал позиционные веса тұздық: из
винрейтов в 533 тысячах партий получилась таблица `TUZDYK_VALUE` в
[`eval.rs:12`](../engine/src/eval.rs#L12) — см.
[05-evaluation.md](05-evaluation.md#веса).

Результаты анализа: [`GAME_ANALYSIS.md`](../GAME_ANALYSIS.md) (35 КБ).

---

# Практические выводы из отчётов

Три вывода, сэкономивших бы много времени, если бы были известны заранее:

**1. Val_loss не предсказывает силу игры.** Сеть Gen3 512×64 имела меньший
val_loss, но играла слабее 256×32. Единственный надёжный критерий — матч.

**2. Разброс по сидам огромен.** K=400, сид 1 → 96.8% побед; тот же K, сид 2 →
72.5%. Один прогон обучения ничего не доказывает. Отсюда
[`train_multiseed.py`](../engine/train_multiseed.py).

**3. Batch size — гиперпараметр первого порядка.** 4096 против 8192 — разница
258 Elo.

Полный разбор — [12-results.md](12-results.md).
