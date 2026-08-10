# 03. Сборка и запуск

## Что уже есть в репозитории и чего нет

Часть бинарных артефактов исключена из git по размеру. Это определяет, что
запустится сразу, а что потребует сборки или обучения.

| Артефакт | В репозитории | Последствие отсутствия |
|----------|---------------|------------------------|
| `engine/nnue_weights.bin` (37 КБ) | **да** | — |
| `engine/opening_book.txt` (1.2 МБ) | **да** | — |
| `web/opening_book.json` (236 КБ) | **да** | — |
| `engine/target/release/togyzkumalaq-engine` | нет | нужна сборка Rust |
| `engine/egtb.bin` (~68 МБ) | нет | движок слабее на ~30 Elo |
| `alphazero-code/browser_model/model.onnx` | нет | уровень «AlphaZero» в браузере не работает |
| `alphazero-code/alphazero/checkpoints/` | нет | нельзя продолжить обучение AlphaZero |

## Требования

| Компонент | Версия |
|-----------|--------|
| Rust | 1.70+ с Cargo |
| Python | 3.10+ (для AlphaZero — 3.11+) |
| Flask | 3.0 |
| PyTorch | 2.0+ с CUDA (только для обучения) |

## Вариант 1: только фронтенд (без сборки)

Быстрее всего запускается автономный браузерный фронтенд из `alphazero-code/`.
AI считается прямо в браузере, бэкенд не нужен.

```bash
cd alphazero-code
python3 -m http.server 8000
```

Открыть <http://localhost:8000>.

Что будет работать: уровни от «Жеңіл» (minimax глубины 2) до «Супер»
(параллельный MCTS на 100 000 симуляций). Уровень «AlphaZero» упадёт — нет
`model.onnx`.

Что будет в консоли: предупреждение `[GameLogger] Server connection: Unavailable`.
Это нормально — логирование партий переключается на `localStorage`
([`game.js:29-52`](../alphazero-code/game.js#L29-L52)).

Страница подтягивает Google Fonts и `onnxruntime-web` с CDN, поэтому без
интернета шрифты будут системными.

## Вариант 2: полный движок

### Сборка

```bash
cd engine
cargo build --release
```

Сборка занимает заметное время из-за `lto = true` и `codegen-units = 1` в
[`Cargo.toml`](../engine/Cargo.toml). Debug-сборка на порядок медленнее и для
игры непригодна.

### Проверка

```bash
cargo test              # юнит-тесты правил и оценки
./target/release/togyzkumalaq-engine perft    # проверка генератора ходов
./target/release/togyzkumalaq-engine bench    # замер скорости поиска
```

### Игра в терминале

```bash
./target/release/togyzkumalaq-engine play
```

### Веб-интерфейс

```bash
pip install flask
python3 web/server.py      # запускать из корня репозитория
```

Открыть <http://localhost:8080>.

Сервер поднимает движок как постоянный дочерний процесс в режиме `serve` и
держит его между ходами — это сохраняет таблицу транспозиций и историю партии
для детекта повторений ([`server.py:27-42`](../web/server.py#L27-L42)).

## Важно: рабочая директория

Движок ищет свои файлы **по относительным путям от текущей директории**
([`main.rs:19-21`](../engine/src/main.rs#L19-L21)):

```rust
const NNUE_PATH: &str = "nnue_weights.bin";
const BOOK_PATH: &str = "opening_book.txt";
const EGTB_PATH: &str = "egtb.bin";
```

Запуск из неправильной директории не выдаст ошибку — движок молча перейдёт на
ручную оценку без книги и таблиц, то есть станет значительно слабее. В логе
появится:

```
No NNUE weights found, using handcrafted eval
```

`web/server.py` подставляет правильный `cwd` явно, при ручном запуске за этим
надо следить самому.

## Генерация недостающих артефактов

### Эндшпильные таблицы

```bash
cd engine
./target/release/togyzkumalaq-engine egtb-gen 4 egtb.bin
./target/release/togyzkumalaq-engine egtb-verify 1000
```

Для `N ≤ 4` получается ~170 млн позиций и файл около 68 МБ. Время генерации
зависит от машины; параметр `max_stones` увеличивать осторожно — объём растёт
комбинаторно.

### Дебютная книга для веб-сервера

```bash
python3 web/generate_book.py
```

Скрипт строит `opening_book.json` из распарсенных партий с фильтрами
`MIN_ELO = 1600` и `MIN_GAMES = 3` ([`generate_book.py:16-17`](../web/generate_book.py#L16-L17)).

### Веса NNUE

Обучение описано в [11-data-and-training.md](11-data-and-training.md).
Боевые веса уже лежат в репозитории, переобучать для запуска не требуется.

## Число потоков

Движок использует все доступные ядра
([`get_num_threads`, main.rs:153](../engine/src/main.rs#L153)):

```rust
std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
```

Переменной окружения для ограничения нет. На слабом сервере (в отчётах упомянут
1 vCPU) это означает однопоточный поиск.

## Деплой

[`deploy.py`](../deploy.py) разворачивает движок и веб на удалённый сервер через
SSH: собирает Rust на сервере, заливает веса, EGTB и книгу, создаёт systemd-юнит.

**Перед использованием прочитайте**
[13-known-issues.md](13-known-issues.md#1-секреты-в-репозитории) — в скрипте
захардкожены боевые учётные данные.

Альтернативные скрипты в `alphazero-code/`: [`deploy.sh`](../alphazero-code/deploy.sh),
[`setup_server.sh`](../alphazero-code/setup_server.sh),
[`add_nginx_config.sh`](../alphazero-code/add_nginx_config.sh) — с описаниями в
[`DEPLOY.md`](../alphazero-code/DEPLOY.md) и [`QUICK_DEPLOY.md`](../alphazero-code/QUICK_DEPLOY.md).
