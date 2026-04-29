# Финальный отчёт: MCTS проект для Тоғызқұмалақ

**Дата:** Apr 19, 2026

## Цель проекта

Создать ИИ сильнее текущего NNUE engine (Gen7, ~1200 Elo над HCE), в идеале уровня чемпионов (2400+ Elo на PlayOK).

## Что достигнуто

### 1. Критичный баг найден и исправлен ✅

**Проблема**: Opening book (21K в engine + 872 в web) содержал 99% mid/endgame позиций. Engine возвращал `depth=0, time=0` — без поиска — для ходов 19-80 если позиция случайно была в книге.

**Симптом**: Чемпионы жаловались "играет хорошо до 18 хода, потом тупит".

**Фикс**: Ограничил book только начальными позициями (stones_played < 20). Применено к engine book.rs и web/server.py.

**Результат**: Engine теперь делает полный поиск (depth 16-22, 1-5 сек/ход) всю партию.

### 2. Инфраструктура ✅

- **Rust MCTS engine** с GPU batch inference (0.9 ms/batch)
- **Clean datagen pipeline** — best_move + eval записываются
- **Master training pipeline** с AMP, warmup, cosine LR, label smoothing
- **Multiple model sizes** — large2m (2.2M), large3m (3.6M), large5m (5.9M)

### 3. Обученные модели

| Модель | Данные | Val accuracy |
|--------|--------|-------------|
| supervised_fresh | PlayOK 1500+ (500K) | **68.7%** human moves |
| distilled (old) | Engine contaminated (1.4M) | 49.4% engine moves |
| **hybrid** | Engine clean (1.4M) + PlayOK (500K) | **52.4%** engine moves |
| large3m (from scratch) | Big clean (2.3M) | 47.9% |
| **large2m_big** (from hybrid init) | Big clean (2.3M) | **52.1%** |

### 4. Data pipeline

- **1.79M чистых позиций** от Gen7 engine (depth 12 self-play)
- **500K PlayOK позиций** от игроков 1500+
- Правильные метки: best_move + eval + game_result
- Все без book pollution

## Что НЕ удалось

### 1. Победить engine ❌

Все модели в прямых играх vs Gen7 engine: **0% винрейт** даже с deep MCTS 400 sims.

| Eval method | Winrate |
|-------------|---------|
| 1-ply lookahead | 2.5-5% |
| Deep MCTS 100 sims | 0% |
| Deep MCTS 200 sims | 0-5% |
| Deep MCTS 400 sims | 0% |

### 2. Плато на 52% ❌

Val accuracy не растёт выше 52% при predicting engine moves:
- 1.4M позиций: 52.4%
- 2.3M позиций: 52.1%
- large3m (больше модель): 47.9% without init

**Выводы:** Data scaling не работает, model scaling без proper init не работает.

## Корневые причины

### Почему 52% потолок на engine moves

1. **Шум в лейблах** — engine best move зависит от search randomness, не уникален
2. **Множество хороших ходов** — в mid/endgame часто 2-3 move имеют одинаковую силу
3. **Архитектура ограничена** — 2M params не может запомнить все паттерны тоғызқұмалақ

### Почему 0% vs engine

1. **Search depth gap** — engine alpha-beta depth 16-22, наш MCTS 200-400 sims ≈ depth 4-6
2. **Value head качественнее, но недостаточно** — 52% policy accuracy × weak search ≈ loss
3. **First-move advantage** — в тоғызқұмалақ белые доминируют при равных силах
4. **1 vCPU deployment** не может крутить deep MCTS 600+ sims

## Что реально работало

✅ **Опенинг book fix** — чемпионы жалоб должно больше не быть на "тупит после 18"
✅ **supervised pretrain** на PlayOK 1500+ даёт 68.7% accuracy на человеческих ходах
✅ **Hybrid training** — engine + PlayOK лучше чем engine only (52.4% vs 49.4%)
✅ **Rust MCTS infrastructure** — production-ready для future GPU deploy

## Реальные перспективы

### Достижимо на текущем железе
- Поддержка NNUE engine с фиксом (должен быть сильным теперь)
- Возможно победить NNUE на локальном GPU с 2000+ sims MCTS + 2M модель
- Но на 1 vCPU сервере MCTS не играбелен на deep search

### Недостижимо без новых ресурсов
- Победить Gen7 engine в deploy конфигурации
- Дотянуть до 2400+ Elo на PlayOK
- Конкурировать с mcts@2541

### Требуется для прорыва
1. **GPU сервер** ($50-200/мес) — для deploy deep MCTS 600+ sims
2. **Намного больше тренировочных данных** — 10M+ позиций от depth 16 engine
3. **Другая архитектура** — transformer/attention для game tree
4. **Время** — 2-4 недели GPU training

## Текущий state

- **Рабочий Engine**: Gen7 NNUE с исправленным book (играет сильно всю партию)
- **Лучшая NN**: large2m_big / hybrid (52.1-52.4% val acc, ~2-5% vs engine)
- **Чистые данные**: 1.8M позиций + 500K PlayOK
- **Полный код**: все в git (commit 4f8f65f)

## Рекомендация

**Тактика**: Использовать NNUE Gen7 с фиксом как primary engine. Он действительно сильный.

**Долгосрочная стратегия**: Если нужна победа над самим Gen7 — ожидайте GPU deploy. Обучение на текущем железе не даст прорыва без изменения parameters deployment.
