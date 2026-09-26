# Документация 9Qumalaqv2

Техническая документация проекта — AI для игры **Тоғызқұмалақ**.

Документация описывает то, что реально находится в коде на момент написания
(коммит `a426934`). Там, где код расходится с отчётами в корне репозитория
(`REPORT.md`, `FULL_REPORT.md`), приоритет отдан коду, а расхождение отмечено явно.

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
| [14-improvement-plan.md](14-improvement-plan.md) | **Аудит: найденные ошибки, их эффект в Elo и план улучшений** |

## С чего начать

- **Хочу запустить** → [03-build-and-run.md](03-build-and-run.md)
- **Хочу понять, как играет движок** → [04-search.md](04-search.md) + [05-evaluation.md](05-evaluation.md)
- **Хочу обучить свою сеть** → [11-data-and-training.md](11-data-and-training.md)
- **Хочу усилить движок** → [14-improvement-plan.md](14-improvement-plan.md)
- **Хочу поправить баги** → [13-known-issues.md](13-known-issues.md)

## Важно перед работой с репозиторием

В [`deploy.py`](../deploy.py) в открытом виде лежит root-пароль от боевого сервера,
и он присутствует в истории коммитов. Подробности и порядок действий —
[13-known-issues.md](13-known-issues.md#1-секреты-в-репозитории).
