# Тоғызқұмалақ - Game Logger Server

Сервер для логирования игровых данных.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python3 server.py
```

Сервер запустится на `http://0.0.0.0:5000`

## API Endpoints

### Health Check
```
GET /api/health
```
Проверка доступности сервера.

### Сохранение игры
```
POST /api/games
Content-Type: application/json

{
  "id": "game_id",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "mode": "pvp" | "bot",
  "aiLevel": "hard",
  "moves": [...],
  "states": [...],
  "aiEvaluations": [...],
  "result": {
    "winner": "white" | "black" | "draw",
    "finalScore": {"white": 82, "black": 80},
    "totalMoves": 150,
    "duration": 3600000
  }
}
```

### Получение игр
```
GET /api/games
```
Возвращает все игры за сегодня.

### Статистика
```
GET /api/games/stats
```
Возвращает статистику по всем играм.

### Экспорт
```
GET /api/games/export
```
Экспорт всех игр в JSON формате.

## Данные

Игровые данные сохраняются в папке `game_logs/` в формате:
- `games_YYYY-MM-DD.json` - игры за конкретный день

## CORS

Сервер настроен для работы с фронтендом на любом порту (CORS включен).

