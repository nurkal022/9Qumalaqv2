# Монореп: реструктуризация (big-bang) — дизайн

- **Дата:** 2026-05-31
- **Статус:** одобрен (дизайн), ожидает вычитки спеки → плана реализации
- **Тип:** организационная реструктуризация существующего монорепозитория

## 1. Контекст и проблема

В одном репозитории смешаны три разные по природе вещи с разным ритмом жизни:

- **Продукт** (`web-v2/`) — то, что видят игроки; должен быть стабильным.
- **Ядро** (Rust-движок `engine/`) — то, что шипится в продукт.
- **Исследование** (`rust-mcts/` + Python-скрипты, чекпойнты, eval, селфплей) — грязный, быстрый поток с множеством экспериментов.

Текущий уровень игры не устраивает; впереди **много экспериментов** по его улучшению. Сейчас это превращается в свалку: правила игры (`board.rs`) скопированы байт-в-байт между движком и MCTS, нет Cargo-workspace, ~5 почти одинаковых train-скриптов, разовые `.sh`, чекпойнты и eval-результаты разбросаны по корню и `rust-mcts/`, артефакты не воспроизводимы и не сравнимы.

## 2. Принятые решения

1. **Топология:** один монореп с чистыми зонами (НЕ разделение на отдельные репозитории).
2. **Глубина:** переложить по зонам **+** ввести Cargo-workspace с общим `core`-crate (конец копиям `board.rs`). Слияние train-скриптов в один пайплайн — **отложено** (отдельная работа потом).
3. **Артефакты:** живут в репо под `research/runs/<дата-имя>/`, **gitignored**; трекаются только `config.yaml` + `summary.md`.
4. **Миграция:** **big-bang** — один заход до чистого состояния, на изолированной ветке, мердж только по «гейту зелени».
5. **Дефолты (не уточнялись, выбраны по идиоматике):** Rust-crate'ы плоско в корне (`core/`, `engine/`, `mcts/`); имя продуктовой зоны — `product/`.

## 3. Целевая структура

```
9QumalaqV2/
├── Cargo.toml              # workspace-корень (members: core, engine, mcts)
│
├── core/                   # crate togyzkumalaq-core: ПРАВИЛА ИГРЫ (единственный board.rs)
│   ├── src/                #   Board, Side, GameResult, UndoInfo, make/unmake,
│   │                       #   valid_moves, парсинг/энкодинг позиции
│   └── Cargo.toml
├── engine/                 # классический движок (eval, search, nnue, egtb, texel, book, tt…)
│   ├── src/                #   board.rs удалён → use togyzkumalaq_core
│   └── Cargo.toml
├── mcts/                   # (бывш. rust-mcts) AlphaZero MCTS
│   ├── src/                #   board.rs удалён → use togyzkumalaq_core
│   └── Cargo.toml
│
├── product/                # то, что шипится игрокам (СТАБИЛЬНОЕ)
│   └── web/                #   (бывш. web-v2: backend + frontend)
│
├── research/               # вся экспериментальная работа (КОД)
│   ├── training/           #   train_*.py (пока как есть; дедуп — отдельно)
│   ├── data/               #   подготовка данных, энкодинг, сбор PlayOK
│   ├── eval/               #   eval-харнессы (.py + .sh)
│   ├── configs/            #   шаблоны конфигов экспериментов        (в git)
│   └── runs/               #   АРТЕФАКТЫ ЭКСПЕРИМЕНТОВ (gitignored)
│       └── 2026-05-31-<имя>/
│           ├── config.yaml #     (в git — что именно запускали)
│           ├── summary.md  #     (в git — результат, eval-числа)
│           ├── checkpoints/#     (gitignored)
│           ├── logs/       #     (gitignored)
│           └── games/      #     (gitignored — селфплей)
│
├── models/                 # «ЧЕМПИОНЫ»: благословлённые бинари/сети (в git)
│   ├── engine/             #   боевой бинарь (baseline) — на него смотрит product
│   └── nets/               #   промотированные ONNX/pt
│
├── datasets/               # крупные общие датасеты (gitignored)
├── tools/                  # deploy_lan.py и операционные скрипты
├── docs/
└── archive/                # заморожённое старьё (старый web/, old-impls)
```

## 4. Карта переездов

| Сейчас | Станет | Как |
|---|---|---|
| `web-v2/` | `product/web/` | `git mv` |
| `rust-mcts/src/` | `mcts/src/` | `git mv` + новый `mcts/Cargo.toml` |
| `rust-mcts/scripts/train_*.py` | `research/training/` | `git mv` |
| `rust-mcts/scripts/{collect_master_games,export_onnx,*encode*}.py` | `research/data/` | `git mv` |
| `rust-mcts/scripts/eval_*.py`, `rust-mcts/run_eval_*.sh`, `play_*.sh` | `research/eval/` | `git mv` |
| `rust-mcts/checkpoints_*`, `eval_results*.txt`, `final_eval.txt` | `research/runs/_legacy/` | `git mv`, затем gitignore |
| `engine/target/release/togyzkumalaq-engine-baseline` | `models/engine/baseline` | `git mv`/`cp` (это благословлённый бинарь) |
| лучший чекпойнт `iter_2645.pt` | `models/nets/iter_2645.pt` | `git mv` (или LFS) |
| `web/` (старый) | `archive/web-old/` | `git mv` |
| `deploy_lan.py` | `tools/deploy_lan.py` | `git mv` |
| `nigtht_report.md`, `CHAMPION_SETUP.md` | `docs/` | `git mv` |
| `6666.apk` (39 МБ) | убрать из репо (в `tools/` gitignored или удалить) | решить при реализации |
| `archive/`, симлинки `game-pars`, `alphazero-code` | на месте; симлинки перенацелить | — |

> `engine/` остаётся `engine/` (просто становится членом workspace без `board.rs`).

## 5. Rust workspace + `core`

- Корневой `Cargo.toml`: `[workspace] members = ["core","engine","mcts"]`, `resolver = "2"`.
- `core/` = crate **`togyzkumalaq-core`**. Содержимое = текущий `board.rs` (он уже починен в этой сессии: фикс `unmake_move` + тесты сохранения камней/симметрии). Публичный API: `Board, Side, GameResult, UndoInfo`, `make_move/unmake_move`, `valid_moves_*`, `is_valid_move`, парсинг/энкодинг строки позиции `w../b../k,k/t,t/side`.
- `engine/` и `mcts/` удаляют свой `board.rs`, добавляют `togyzkumalaq-core = { path = "../core" }`, заменяют `mod board;` на `use togyzkumalaq_core::...`.
- Парсинг позиции (`parse_position`) сейчас живёт в `engine/src/main.rs` и в `rust-mcts/src/main.rs` (две копии) — переносится в `core` как единственный источник.

## 6. Артефакты, `.gitignore`, конвенция эксперимента

`.gitignore` (ключевое):
```
target/
**/node_modules
**/__pycache__
**/dist
*.db
datasets/**
research/runs/**
!research/runs/**/config.yaml
!research/runs/**/summary.md
```
- `models/**` — трекается (baseline ~690 КБ ок). Для тяжёлых `.pt`/`.onnx` — git-LFS как опция (решить при реализации).
- **Конвенция эксперимента:** один запуск = один каталог `research/runs/<дата>-<имя>/`. Resolved-конфиг кладётся внутрь (`config.yaml`, трекается), чекпойнты/логи/селфплей-партии пишутся внутрь (gitignored), в конце пишется `summary.md` с eval-числами (трекается). Сам лаунчер — это реализация (план), не спека; здесь фиксируется только конвенция каталога.

## 7. Развязка продукта (`models/`)

- `product/web/backend/app/config.py`: `engine_path` → `models/engine/baseline` (вместо `engine/target/release/...`). Продукт берёт движок **только** из `models/`, не из build-вывода исследований. Это финализирует уже сделанное в этой сессии (переключение на baseline) и развязывает продукт от грязи экспериментов.
- Промоушен: когда эксперимент «побеждает», его бинарь/сеть копируется в `models/`; продукт переключается изменением одной строки/env (`ENGINE_PATH`).

## 8. Чек-лист поломок путей (big-bang)

1. **`REPO_ROOT` в бэкенде:** `web-v2` → `product/web` меняет глубину (`Path(__file__).parents[3]` → `parents[4]`). Починить.
2. **Захардкоженные абсолютные пути** в train/eval-скриптах (`~/9QumalaqV2/...`, `rust-mcts/...`) → перенацелить или вынести в конфиг.
3. **Симлинки** `alphazero-code`, `game-pars` (нужны сборке/обучению) — перенацелить относительно новой раскладки.
4. **Пути в** `deploy_lan.py` (→ `tools/`), любых CI/`.sh`, `play_*.sh`.
5. **`.gitignore`** обновляется ДО переездов (чтобы не закоммитить артефакты).
6. **Cargo:** `engine/Cargo.toml`, новый `mcts/Cargo.toml`, корневой workspace; имя пакета rust-mcts → `mcts`.

## 9. Исполнение big-bang + гейт зелени

- Изолированная ветка `restructure` (worktree).
- Все перемещения через `git mv` (история сохраняется).
- Порядок внутри одного захода: (1) `.gitignore`; (2) workspace + `core` + дедуп `board.rs`/`parse_position`; (3) переезды зон по карте; (4) починка всех путей; (5) развязка продукта на `models/`.
- **Гейт зелени (merge-блокер):**
  - `cargo build` и `cargo test` для core + engine + mcts;
  - фронт: `tsc --noEmit`, `vitest run`, `vite build`;
  - бэкенд: `pytest` (с движком из `models/engine/baseline`);
  - smoke: дуэль движка через serve (1-2 партии) — протокол жив.
- Один PR, мердж только при полной зелени.

## 10. Риски и откат

- **Риск:** big-bang большой дифф, легко пропустить захардкоженный путь → продукт не стартует. **Снижение:** гейт зелени покрывает бэкенд/фронт/движок; ничего не мерджится «вслепую».
- **Риск:** перемещения ломают историю/симлинки. **Снижение:** `git mv`, явная перенастройка симлинков в чек-листе.
- **Откат:** удалить ветку `restructure` — main/рабочая ветка нетронута.

## 11. Вне scope (отложено)

- Слияние 5 train-скриптов в один параметризуемый пайплайн + единый модуль данных/энкодинга.
- Разбиение god-файлов (`search.rs` 1224, `main.rs` 942).
- Синхронизация `texel.rs ↔ eval.rs`, дедуп `predict_landing`.
- Лаунчер экспериментов (реализация конвенции из §6).
- git-LFS (решается отдельно, если нужно).

## 12. Критерии готовности (acceptance)

- Корневой workspace собирается; `board.rs` существует в единственном экземпляре (в `core`); `diff` дублей невозможен.
- Зоны соответствуют §3; артефакты gitignored по §6.
- Продукт берёт движок из `models/engine/baseline`; бэкенд-тесты зелёные.
- Все проверки из гейта §9 проходят.
- README/доки указывают на новую раскладку.
