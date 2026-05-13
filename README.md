# 9Qumalaqv2

AI-движок и веб-демо для игры **Тоғызқұмалақ**. Проект объединяет Rust-движок с alpha-beta поиском, NNUE-оценкой, генерацией обучающих данных, анализом партий и браузерным интерфейсом для игры против AI.

## Возможности

- Полная логика игры Тоғызқұмалақ: ходы, захваты, тұздық, завершение партии.
- Rust-движок `togyzkumalaq-engine` с alpha-beta поиском, iterative deepening, TT, killer/history heuristics и quiescence search.
- Поддержка NNUE-весов, handcrafted evaluation, эндшпильных таблиц и дебютной книги.
- Генерация self-play данных и пайплайны обучения/дообучения NNUE.
- Python-скрипты для парсинга, анализа и экспорта позиций из партий.
- Flask-веб-сервер с постоянным процессом движка и HTML-интерфейсом.

## Структура проекта

```text
.
├── engine/                 # Rust-движок, NNUE, поиск, datagen
│   ├── src/                # Исходный код движка
│   ├── Cargo.toml          # Rust-манифест
│   ├── nnue_weights.bin    # Бинарные веса NNUE
│   └── opening_book.txt    # Дебютная книга для движка
├── web/                    # Веб-интерфейс и Flask API
│   ├── index.html
│   ├── server.py
│   └── opening_book.json
├── alphazero-code/         # Эксперименты AlphaZero и отдельное веб-демо
├── parsed_games/           # Распарсенные партии
├── gameNew2/               # Исходные текстовые партии
├── analyze_games.py        # Анализ партий
├── parse_games.py          # Парсер партий
├── export_positions.py     # Экспорт позиций
└── *.md                    # Технические отчеты и результаты экспериментов
```

## Требования

- Rust 1.70+ с Cargo
- Python 3.10+
- Flask для веб-сервера

Для веб-демо установите Python-зависимости:

```bash
pip install -r alphazero-code/requirements.txt
```

## Быстрый старт

Соберите движок:

```bash
cd engine
cargo build --release
```

Запустите интерактивную игру в терминале:

```bash
./target/release/togyzkumalaq-engine play
```

Запустите веб-интерфейс:

```bash
cd ..
python3 web/server.py
```

После запуска откройте:

```text
http://localhost:8080
```

## Команды движка

Все команды запускаются из каталога `engine` после сборки:

```bash
./target/release/togyzkumalaq-engine play
./target/release/togyzkumalaq-engine bench
./target/release/togyzkumalaq-engine perft
./target/release/togyzkumalaq-engine selfplay
./target/release/togyzkumalaq-engine texel
./target/release/togyzkumalaq-engine match [games] [time_ms]
./target/release/togyzkumalaq-engine match-nnue <weights_a> <weights_b> [games] [time_ms]
./target/release/togyzkumalaq-engine datagen [games] [depth] [threads] [prefix]
./target/release/togyzkumalaq-engine egtb-gen [max_stones] [output]
./target/release/togyzkumalaq-engine egtb-verify [tests]
./target/release/togyzkumalaq-engine analyze <position> [time_ms]
./target/release/togyzkumalaq-engine serve
```

Формат позиции для `analyze`:

```text
w0,w1,...,w8/b0,...,b8/kw,kb/tw,tb/side
```

Пример:

```bash
./target/release/togyzkumalaq-engine analyze "9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0" 3000
```

## Работа с данными

Парсинг партий:

```bash
python3 parse_games.py
```

Анализ партий:

```bash
python3 analyze_games.py
python3 analyze_deep.py
```

Экспорт позиций:

```bash
python3 export_positions.py
```

Генерация данных для NNUE:

```bash
cd engine
./target/release/togyzkumalaq-engine datagen 10000 8 4 local
```

## Веб API

`web/server.py` поднимает Flask API и держит Rust-движок в постоянном процессе, чтобы сохранялись таблица транспозиций и история партии.

Основные endpoint'ы:

- `GET /` — HTML-интерфейс.
- `POST /api/move` — получить ход движка.
- `POST /api/newgame` — сбросить состояние новой партии.
- `POST /api/position` — передать текущую позицию в историю движка.

## Отчеты и документация

- `REPORT.md` — технический отчет по архитектуре и оценке.
- `FULL_REPORT.md` — полный отчет по проекту.
- `GAME_ANALYSIS.md` — анализ партий.
- `NNUE_EXPERIMENTS.md` — эксперименты с NNUE.
- `ALPHAZERO_INTEGRATION.md` — интеграция AlphaZero-подхода.
- `alphazero-code/alphazero/README.md` — документация по AlphaZero-модулю.

## Примечания

- Для максимальной силы игры запускайте release-сборку: `cargo build --release`.
- Файлы `nnue_weights.bin`, `opening_book.txt` и `opening_book.json` используются движком и веб-сервером, если находятся в ожидаемых каталогах.
- Скрипты развертывания могут содержать окружение-зависимые параметры. Перед использованием проверьте их и вынесите секреты в переменные окружения.
