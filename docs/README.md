# Документация 9Qumalaqv2

Техническая документация проекта — AI для игры **Тоғызқұмалақ**.

Документация описывает то, что реально находится в коде на момент написания
(коммит `a426934`). Там, где код расходится с отчётами в корне репозитория
(`REPORT.md`, `FULL_REPORT.md`), приоритет отдан коду, а расхождение отмечено явно.

> **Структура изменилась после написания этих документов** (реструктуризация в
> монорепозиторий, май 2026). Старые пути теперь такие:
>
> | В документах | Сейчас |
> |---|---|
> | `engine/` | `engine/` (урезан; правила вынесены в `core/`) |
> | `web/` (Flask) | `archive/web-old/`; новый продукт в `product/web/` |
> | `alphazero-code/` | `mcts/` (Rust) + `research/training/`; старый Python-код в облаке (`archive-old-impls`) |
> | `deploy.py`, `parse_games.py`, `analyze_*.py`, `export_positions.py` | `archive/misc/` |
> | `REPORT.md`, `FULL_REPORT.md`, `*_REPORT.md` | `archive/reports/` |
> | `parsed_games/`, `gameNew2/`, данные | облако, см. раздел «Данные» в [корневом README](../README.md#данные) |
>
> Описание алгоритмов (поиск, оценка, NNUE, EGTB) остаётся актуальным.

## Навигация

| Документ | Содержание |
|----------|-----------|
| [01-overview.md](01-overview.md) | Обзор проекта, карта репозитория, два трека AI |
| [02-game-rules.md](02-game-rules.md) | Правила тоғызқұмалақ и их отражение в коде |
| [03-build-and-run.md](03-build-and-run.md) | Сборка, запуск, необходимые артефакты |
| [04-search.md](04-search.md) | Alpha-beta поиск: отсечения, расширения, SMP, тайм-менеджмент |
| [05-evaluation.md](05-evaluation.md) | Ручная оценка (HCE), Texel-тюнинг, эндшпильные коррекции |
| [06-nnue.md](06-nnue.md) | NNUE: архитектура, признаки, бинарный формат, инференс |
| [07-egtb-and-book.md](07-egtb-and-book.md) | Эндшпильные таблицы и дебютная книга |
| [08-cli-and-protocol.md](08-cli-and-protocol.md) | CLI-команды и stdin/stdout протокол `serve` |
| [09-web.md](09-web.md) | Два веб-фронтенда и HTTP API |
| [10-alphazero.md](10-alphazero.md) | Трек AlphaZero: сеть, MCTS, обучение |
| [11-data-and-training.md](11-data-and-training.md) | Данные, парсинг партий, пайплайн обучения NNUE |
| [12-results.md](12-results.md) | Результаты экспериментов и прогресс по Elo |
| [13-known-issues.md](13-known-issues.md) | Известные проблемы и технический долг |

## С чего начать

- **Хочу запустить** → [03-build-and-run.md](03-build-and-run.md)
- **Хочу понять, как играет движок** → [04-search.md](04-search.md) + [05-evaluation.md](05-evaluation.md)
- **Хочу обучить свою сеть** → [11-data-and-training.md](11-data-and-training.md)
- **Хочу поправить баги** → [13-known-issues.md](13-known-issues.md)

## Важно перед работой с репозиторием

В истории коммитов есть root-пароли серверов (из текущего кода они удалены,
`archive/misc/deploy.py` берёт пароль из `DEPLOY_PASSWORD`). Эти пароли надо сменить. Подробности и порядок действий —
[13-known-issues.md](13-known-issues.md#1-секреты-в-репозитории).
