# 09. Веб-интерфейсы

В проекте **два независимых фронтенда**. Они не связаны, используют разные
реализации правил и разные AI. Выбор между ними зависит от того, нужен ли
полноценный движок.

| | `web/` | `alphazero-code/` |
|---|--------|-------------------|
| Требует бэкенд | **да** | нет |
| Где считается AI | Rust-движок на сервере | браузер пользователя |
| Сила игры | максимальная | средняя |
| Язык интерфейса | казахский | казахский |
| Режим анализа | есть | нет |
| Логирование партий | нет | есть (сервер + localStorage) |
| Уровни сложности | время на ход (1–10 с) | 8 уровней AI |
| Правила игры | Rust `board.rs` | JS `game.js` |

---

# Фронтенд 1: `web/`

Файлы: [`web/index.html`](../web/index.html) (39 КБ, всё в одном файле),
[`web/server.py`](../web/server.py)

## Запуск

```bash
pip install flask
python3 web/server.py          # из корня репозитория
```

<http://localhost:8080>

## Архитектура

```
Браузер ──HTTP──> Flask (server.py) ──stdin/stdout──> togyzkumalaq-engine serve
   │                    │                                    │
   │              opening_book.json                   nnue_weights.bin
   │              (вероятностный выбор)               opening_book.txt
   │                                                  egtb.bin
   └── дублирует правила игры на JS для отрисовки и валидации
```

Ключевое архитектурное решение: движок живёт **постоянным процессом**, а не
запускается на каждый ход. Это сохраняет между ходами таблицу транспозиций
(следующий поиск стартует с прогретой TT) и историю партии (нужна для детекта
повторений).

## HTTP API

### `GET /`

Отдаёт [`index.html`](../web/index.html).

### `POST /api/move`

Запрос хода движка.

```json
{
  "board": {
    "pits": [[9,9,9,9,9,9,9,9,9], [9,9,9,9,9,9,9,9,9]],
    "kazan": [0, 0],
    "tuzdyk": [-1, -1],
    "side_to_move": 0
  },
  "time_ms": 3000,
  "use_book": true
}
```

Ответ движка:

```json
{
  "bestmove": 6, "score": 142, "depth": 24,
  "nodes": 8451203, "time_ms": 3012, "nps": 2807203
}
```

Ответ из книги (`depth`, `nodes`, `time_ms` нулевые):

```json
{
  "bestmove": 6, "score": 0, "depth": 0, "nodes": 0, "time_ms": 0, "nps": 0,
  "book": { "source": "book", "games": 42, "win_rate": 0.583 }
}
```

Терминальная позиция:

```json
{ "terminal": true, "result": "white_win" }
```

Порядок обработки ([`server.py:139-197`](../web/server.py#L139-L197)):

1. Позиция переводится в строку движка (`board_to_pos`).
2. Если `use_book` — поиск в `opening_book.json`, при попадании мгновенный ответ.
3. Иначе `go time <ms> pos <позиция>` движку под мьютексом.
4. Разбор строки ответа в JSON.

### `POST /api/newgame`

Сброс состояния движка: TT, эвристики, история партии.

### `POST /api/position`

Добавление позиции в историю движка. Клиент вызывает **после каждого хода
человека** ([`index.html:1093-1098`](../web/index.html#L1093-L1098)) — без этого
движок не увидит повторения.

После хода самого движка вызывать не нужно: он добавляет получившуюся позицию
сам ([`main.rs:760-764`](../engine/src/main.rs#L760-L764)).

## Интерфейс

### Лобби

Перед партией выбираются ([`index.html:499-543`](../web/index.html#L499-L543)):

- **Сторона**: белые или чёрные
- **Время на ход**: 1, 3 (по умолчанию), 5 или 10 секунд
- **Режим**: чистая игра или анализ

### Игровой экран

| Элемент | Описание |
|---------|----------|
| Доска | анимированная раздача камней ([`animateMove`](../web/index.html#L854)) |
| Список ходов | нотация с пометкой источника: `book` или `engine` |
| Панель анализа | ход, оценка, глубина + раскрываемые узлы/nps/время |
| Артқа | откат на два полухода |
| Талдау | перемотка партии по ходам с возвратом к текущей позиции |

### Дублирование правил

Фронтенд реализует правила игры **на JavaScript** для отрисовки, анимации и
валидации кликов ([`makeMove`](../web/index.html#L768),
[`isValidMove`](../web/index.html#L827),
[`getGameResult`](../web/index.html#L832)). Это третья независимая реализация
правил в проекте после Rust и `game.js` — расхождение здесь приведёт к тому, что
браузер и движок будут видеть разные позиции.

### Особенность отката

[`undoMove`, index.html:1147](../web/index.html#L1147)

Откат снимает два полухода, пересчитывает доску с начала партии и вызывает
`/api/newgame`. Последнее **полностью очищает историю партии в движке**, не
восстанавливая её для оставшихся ходов. После отката детект повторений на
предыдущих позициях перестанет срабатывать до конца партии. На силу игры влияет
слабо, но поведение неочевидное.

---

# Фронтенд 2: `alphazero-code/`

Файлы: [`index.html`](../alphazero-code/index.html),
[`game.js`](../alphazero-code/game.js) (63 КБ),
[`mcts-worker.js`](../alphazero-code/mcts-worker.js),
[`styles.css`](../alphazero-code/styles.css)

## Запуск

Бэкенд не нужен — достаточно любого статического сервера:

```bash
cd alphazero-code
python3 -m http.server 8000
```

<http://localhost:8000>

## Уровни AI

[`AI_LEVELS`, game.js:1103](../alphazero-code/game.js#L1103)

| Уровень | Название | Алгоритм | Параметры |
|---------|----------|----------|-----------|
| easy | Жеңіл | minimax | глубина 2 |
| medium | Орташа | minimax | глубина 4 |
| hard | Қиын | minimax | глубина 6 |
| expert | Эксперт | MCTS | 5 000 симуляций / 2 с |
| master | Мастер | MCTS | 15 000 / 5 с |
| grandmaster | Гроссмейстер | MCTS | 30 000 / 10 с |
| super | Супер | параллельный MCTS | 100 000 / 30 с |
| alphazero | AlphaZero | ONNX + MCTS | 200 симуляций / 10 с |

Реализации: [`MinimaxAI`](../alphazero-code/game.js#L722) (alpha-beta с
сортировкой ходов), [`MCTSAI`](../alphazero-code/game.js#L548) (UCT с
`C = 1.41`), [`ParallelMCTSAI`](../alphazero-code/game.js#L657) (через
Web Worker), [`AlphaZeroAI`](../alphazero-code/game.js#L895) (батчевый MCTS
с нейросетевой оценкой).

Уровень «AlphaZero» **не работает** — `browser_model/model.onnx` отсутствует в
репозитории. Загружаются только `alphazero-inference.js` и `metadata.json`.

## Логирование партий

[`GameLogger`, game.js:7](../alphazero-code/game.js#L7)

Класс пытается отправлять сыгранные партии на сервер для сбора обучающих данных.
Адрес определяется автоматически ([`game.js:14-24`](../alphazero-code/game.js#L14-L24)):

```javascript
if (hostname === 'localhost' || hostname === '127.0.0.1') {
    this.serverUrl = 'http://localhost:5000/api';
} else {
    // на проде — с учётом возможного подпути /togyzqumalaq
}
```

При недоступности сервера (таймаут 2 секунды) логирование прозрачно
переключается на `localStorage`. В консоли появится:

```
[GameLogger] Server connection: Unavailable, using localStorage fallback
```

Это **не ошибка** — игра полностью работоспособна без сервера.

Опциональный сервер логирования — [`alphazero-code/server.py`](../alphazero-code/server.py),
эндпоинты: `/api/health`, `/api/games`, `/api/games/stats`, `/api/games/export`.
Описание — [`README_SERVER.md`](../alphazero-code/README_SERVER.md).

## Внешние зависимости

Страница подключает два ресурса с CDN
([`index.html:8-10, 198`](../alphazero-code/index.html#L8-L10)):

- Google Fonts (Noto Sans)
- `onnxruntime-web` для уровня AlphaZero

Без интернета шрифты будут системными, остальное работает.

---

# Деплой

Основной скрипт — [`deploy.py`](../deploy.py): собирает движок на сервере,
заливает веса, EGTB, книгу и веб-файлы, создаёт systemd-юнит.

Альтернативные скрипты в `alphazero-code/`: [`deploy.sh`](../alphazero-code/deploy.sh),
[`setup_server.sh`](../alphazero-code/setup_server.sh),
[`upload_files.sh`](../alphazero-code/upload_files.sh),
[`add_nginx_config.sh`](../alphazero-code/add_nginx_config.sh).

**Перед использованием** прочитайте
[13-known-issues.md](13-known-issues.md#1-секреты-в-репозитории) — в `deploy.py`
захардкожены боевые учётные данные.
