# AlphaZero → 3000 Elo: кампания на большом сервере

Ветка `alphazero`. Цель: сеть с поиском MCTS уровня ~3000 Elo, которая обыгрывает
чемпионов. В `main` (стабильная версия) изменения попадают только после
внешнего гейта (см. «Как мерить»).

## Где мы сейчас (сентябрь 2026)

| Что | Результат |
|---|---|
| Лучший bootstrap: `models/nets/phase1/sup1500.pt` (supervised, партии PlayOK ≥1500, 2 млн позиций, `large2m` = 10×192) | **43.8% @1-ply** и 37.5% @200 sims против нашего alpha-beta `baseline` |
| Старый bootstrap `models/nets/mcts_2026-06-18/d2.pt` | 17% @200 sims |
| Gated self-play (`train_loop.py`, keep-best) | деградаций больше нет, но и роста нет: все кандидаты отклонены |
| Конкурент 9qum.com (AlphaZero + score-head) | упёрся в ту же стену: 5/5 итераций отклонено при 43–47% |

**Главная стена — value, а не policy.** Что уже проверено и не сработало
(не повторять):
- дистилляция value из статической оценки движка: 37.5% → 15.6% @200 (это та же функция, по которой ищет baseline);
- value по исходам партий движок-движок: 5.0% @200;
- общий trunk для сильной policy и чистой value: policy разваливается;
- self-play от слабого bootstrap: цели обучения не лучше собственной policy, сигнала улучшения нет.

## Порядок работ

0. **Правило конца партии.** Настоящее правило: игра кончается, когда **у того, кто
   ходит**, нет камней (проверено на 110 партиях PlayOK и 1120 позициях 9qum, 0 противоречий).
   В этой ветке `core/` и `research/alphazero/game.py` пока на старом правиле
   «пуста любая сторона». Нужно перенести **только** фикс правила из ветки `endgame-rules`
   (коммиты `rules(py): terminal only when the side to move has no stones` и соответствующий
   коммит в `core/`) без tempo-оценки. Иначе self-play учит value на неправильных исходах.
1. **Отдельная value-сеть** (без общего trunk с policy), чтобы обучение value не ломало policy.
2. **Score-head** (предсказание разницы камней). Победа определяется разницей, а
   насыщающийся win-prob не отличает «+2» от «+20». Это единственная идея 9qum, которой у нас нет.
3. **Масштаб на сервере:** больше сеть (`large3m` и выше), больше sims в self-play
   (`--selfplay-full-mcts --sims 800+`), больше партий на итерацию, value по
   исходам глубокого поиска. На ноутбуке (RTX 5080 laptop) это было невозможно.
4. После каждого шага сначала внутренний гейт против `baseline` (keep-best в `train_loop.py`), затем внешний.

## Запуск на сервере

Нужны NVIDIA GPU с CUDA 12, Rust ≥1.80, Python 3.12, `zstd` и доступ к приватному датасету HF.

```bash
git clone -b alphazero https://github.com/nurkal022/9Qumalaqv2.git && cd 9Qumalaqv2
hf auth login
HF_REPO=<hf-датасет> bash tools/az_server_setup.sh   # сборка, данные, EGTB; в конце печатает export-ы
# выполнить напечатанные export NVIDIA_LIBS=... ORT_DYLIB_PATH=... EXPERT_DIR=...
```

Короткая проверка (несколько минут):

```bash
python3.12 research/training/train_loop.py --init-checkpoint models/nets/phase1/sup1500.pt \
  --iterations 1 --games 8 --sims 50 --eval-sims 50 --eval-pairs 2 --workers 8 \
  --checkpoint-dir research/runs/smoke --log research/runs/smoke.log
```

Основной прогон (параметры подобрать под сервер: `--workers` примерно по числу ядер):

```bash
RUN=research/runs/$(date +%F)-az-server && mkdir -p $RUN
python3.12 research/training/train_loop.py \
  --init-checkpoint models/nets/phase1/sup1500.pt --model-size large2m \
  --selfplay-full-mcts --sims 800 --games 400 --workers 64 \
  --eval-sims 400 --eval-interval 1 --eval-pairs 50 --gate-margin 0 \
  --expert-ratio 0.5 --expert-decay 0.97 --expert-min 0.1 \
  --iterations 500 --checkpoint-dir $RUN --log $RUN/train.log
```

Прогон возобновляется: перезапустите с `--resume <dir>/latest.pt`. На диске хранятся
только `latest.pt`, `best.pt` и временный `candidate.onnx`.

Типичные ловушки:
- Если GPU-процесс висит, все потоки стоят на futex, а процесса нет в `nvidia-smi`,
  значит дочернему процессу не хватает CUDA `LD_LIBRARY_PATH`/`ORT_DYLIB_PATH`.
  Проверьте export-ы.
- ONNX-экспорт только с `dynamo=False` (так уже сделано в `export_onnx`).

## Как мерить

Сначала прочитайте [`MEASUREMENT_PROTOCOL.md`](MEASUREMENT_PROTOCOL.md). Кратко:
- матчи против своей же линейки ничего не доказывают: прошлые «+140…+398 Elo» были переобучением;
- считаются только **парные внешние гейты в одном временном окне** (кандидат и эталон
  играют против одного внешнего соперника: 9qum.com ladder, `tools/9qum/`; PlayOK, `tools/playok/`);
- минимум 40 партий, для решений 100+;
- точность офлайн-оценки не равна силе игры.

Для отметки «3000» нужна подтверждённая победа над ботом 9qum верхнего уровня (ЗМС,
заявлено ~3000) и над сильнейшими людьми PlayOK (2500+) на парных гейтах.
