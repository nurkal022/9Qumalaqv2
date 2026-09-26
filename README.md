# 9Qumalaqv2 — AI для Тоғызқұмалақ

Движки, обучение и веб-продукт для игры **Тоғызқұмалақ**. Монорепозиторий:
общие правила игры на Rust, два движка (alpha-beta + NNUE и AlphaZero-MCTS),
веб-продукт для игроков, исследовательский пайплайн и боты для живой игры
на PlayOK и 9qum.com.

Код лежит в git. Датасеты, чекпоинты и сети лежат в облаке (раздел [Данные](#данные)).

## Карта репозитория

| Путь | Что там |
|---|---|
| [`core/`](core/) | Крейт `togyzkumalaq-core`, **единственный источник правил игры** (доска, ходы, тұздық, конец партии, парсинг позиции). От него зависят оба движка. |
| [`engine/`](engine/) | Классический движок `togyzkumalaq-engine` (Rust): alpha-beta, TT, NNUE-оценка, EGTB, дебютная книга, datagen, Texel-тюнинг, режим `serve` для бэкенда. |
| [`mcts/`](mcts/) | AlphaZero-движок (Rust): MCTS/Gumbel, self-play, league, replay buffer, инференс через ONNX Runtime (CUDA). |
| [`product/web/`](product/web/) | Продукт: `backend/` на FastAPI + Alembic, `frontend/` на React + Vite + TypeScript. Движок берётся из `models/engine/baseline`. |
| [`models/`](models/) | «Утверждённые» артефакты. В git лежат `engine/baseline` (бинарник для продукта), `nnue_weights.bin`, `opening_book.txt`. Тяжёлые сети лежат в облаке. См. [`models/README.md`](models/README.md). |
| [`research/`](research/) | Код экспериментов: `training/` (NNUE v2, AlphaZero, дистилляция, value-head), `data/` (конвертеры PlayOK/9qum, фичи, экспорт ONNX), `eval/` (матчи и оценка). Прогоны пишутся в `research/runs/` (в git только `config.yaml` и `summary.md`). |
| [`tools/`](tools/) | Скрипты: `ab_match.py` (A/B-матчи движков), `deploy_lan.py` (деплой), `playok/` (бот и мост для PlayOK), `9qum/` (сборщик данных и ladder-бот для 9qum.com, см. [`tools/9qum/README.md`](tools/9qum/README.md)), диагностика эндшпиля и калибровка value. |
| [`docs/`](docs/) | Документация: техническое описание `01…13` (см. [`docs/README.md`](docs/README.md)), протокол измерений [`MEASUREMENT_PROTOCOL.md`](docs/MEASUREMENT_PROTOCOL.md), дизайны и планы в `superpowers/specs` и `superpowers/plans`. |
| [`archive/`](archive/) | Замороженная история: старые отчёты (`reports/`), старый Flask-веб (`web-old/`), эксперименты NNUE (`engine-experiments/`), разовые скрипты (`misc/`). В git только код и отчёты, данные архива лежат в облаке. |

Три Rust-крейта объединены в один Cargo workspace ([`Cargo.toml`](Cargo.toml)),
профиль release с LTO задаётся в корне.

## Сборка и тесты

```bash
# Rust (core + engine + mcts)
cargo build --release
cargo test -p togyzkumalaq-core      # правила
cargo test -p togyzkumalaq-engine    # движок
# mcts подгружает ONNX Runtime в рантайме: LD_LIBRARY_PATH должен указывать на nvidia pip-библиотеки.

# Продукт: backend
cd product/web/backend && .venv/bin/python -m pytest -q

# Продукт: frontend
npm --prefix product/web/frontend ci
npm --prefix product/web/frontend run typecheck
npm --prefix product/web/frontend test
npm --prefix product/web/frontend run build
```

Команды движка: `play`, `serve`, `analyze <позиция> [ms]`, `match`, `datagen`,
`egtb-gen`, `texel` и другие. Полный список в
[`docs/08-cli-and-protocol.md`](docs/08-cli-and-protocol.md).

## Продукт

Бэкенд запускает движок из `models/engine/baseline` (переопределяется через
`ENGINE_PATH`). Новый движок попадает в продукт только после того, как выиграл
у `baseline` в serve-дуэли. Затем его копируют поверх `models/engine/baseline`.

## Данные

Всё, что не код, лежит в **приватном** датасете Hugging Face
**`HF_REPO_PLACEHOLDER`** (около 8 ГБ исходных данных, упакованных в `tar.zst` общим объёмом 2.4 ГБ).
Внутри архивов пути идут от корня репозитория, поэтому архив распаковывается прямо в клон.

| Архив | Что внутри | Куда распаковывается |
|---|---|---|
| `playok-games-archive` | ~380k сырых партий PlayOK + извлечённые `.bin` (elo1400/1500/1600, mcts_training) | `archive/datasets/game-pars/` |
| `playok-games-current` | ~183k партий PlayOK, `games.zip`, скрейпер | `game-pars/` |
| `datasets-other` | mergeData, parsed_games, gameNew2, expertsRV, партии из APK и сервера | `archive/datasets/` |
| `9qum-corpus` | корпус 9qum.com: партии, игроки, дебюты, телеметрия их обучения, `train/` | `data/9qum/` |
| `models` | NNUE v2, сети phase1/MCTS, эксперименты с переобучением, EGTB (`egtb.bin`) | `models/` |
| `research-runs` | прогоны обучения (`_legacy` чекпоинты v3/night, `2026-06-08-corrected`), `runs/` | `research/runs/`, `runs/` |
| `archive-mcts-experiments` | ранние чекпоинты и данные AlphaZero | `archive/mcts-experiments/` |
| `archive-old-impls` | старый код `alphazero-code` (Python) | `archive/old-impls/` |
| `misc` | сторонние бинарники (APK, `Togyz/*.exe`) и прочие файлы из корня | корень |

Восстановление:

```bash
pip install -U huggingface_hub zstd   # или системный zstd
hf auth login                          # нужен доступ к приватному репо
hf download HF_REPO_PLACEHOLDER --repo-type dataset --local-dir /tmp/togyz-data
cd /tmp/togyz-data && sha256sum -c SHA256SUMS
for f in *.tar.zst; do tar -I zstd -xf "$f" -C /path/to/9Qumalaqv2; done
```

Можно скачать один архив: `hf download HF_REPO_PLACEHOLDER models.tar.zst --repo-type dataset --local-dir .`

## Ветки

- `main` — **стабильная версия**: правила, движок `baseline`, продукт, NNUE v2 (phase A).
  Сюда попадает только то, что проверено замерами.
- `alphazero` — **кампания AlphaZero до уровня ~3000 Elo** (цель: обыгрывать чемпионов).
  Запускается на большом сервере. Вся разработка MCTS и обучения идёт здесь, в `main`
  изменения попадают только после внешнего гейта. План и запуск описаны в `docs/ALPHAZERO_3000.md` этой ветки.
- `endgame-rules` — исправление правила конца партии, EGTB `TKEGTB02` и tempo-оценка.
  **Не мержить:** на парных внешних замерах вышло слабее. Правило «ход невозможен у того,
  кто ходит» стоит перенести отдельно, без tempo-терма.

## Секреты

Пароли и токены не коммитим. Всё передаётся через переменные окружения:
`DEPLOY_PASSWORD`, `SSH_PASSWORD`, `PLAYOK_USER`/`PLAYOK_PW`, `A_PW`/`B_PW`.
В старой истории репозитория есть пароли серверов: они скомпрометированы и должны быть сменены.

## Как измерять силу

Сначала прочитайте [`docs/MEASUREMENT_PROTOCOL.md`](docs/MEASUREMENT_PROTOCOL.md).
Коротко: матчи внутри своей линейки ничего не доказывают, считаются только парные
внешние гейты в одном временном окне, а точность офлайн-оценки не равна силе игры.
