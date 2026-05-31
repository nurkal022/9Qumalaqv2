from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

I18N = {
    "validation_failed": ("Деректер дұрыс емес", "Невалидные данные"),
    "auth_required": ("Кіру қажет", "Требуется вход"),
    "not_owner": ("Бұл партия сізге тиесілі емес", "Эта партия не принадлежит вам"),
    "not_found": ("Табылмады", "Не найдено"),
    "username_taken": ("Бұл логин бос емес", "Логин занят"),
    "invalid_credentials": ("Логин не пароль қате", "Неверный логин или пароль"),
    "rate_limited": ("Тым жиі. Кейінірек көріңіз.", "Слишком часто. Попробуйте позже."),
    "engine_unavailable": ("Қозғалтқыш қол жетімсіз", "Движок недоступен"),
    "illegal_move": ("Заңсыз жүріс", "Недопустимый ход"),
    "game_not_active": ("Партия аяқталған", "Партия завершена"),
    "internal": ("Серверде қате", "Ошибка сервера"),
}


class AppError(HTTPException):
    def __init__(self, code: str, http: int, details: dict | None = None):
        super().__init__(status_code=http, detail={"code": code, "details": details or {}})
        self.code = code


def _envelope(code: str, details: dict | None = None) -> dict:
    kk, ru = I18N.get(code, ("Қате", "Ошибка"))
    return {"error": {"code": code, "messageKk": kk, "messageRu": ru, "details": details or {}}}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(_, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content=_envelope(exc.code, exc.detail.get("details")))

    @app.exception_handler(ValidationError)
    async def validation(_, exc: ValidationError):
        return JSONResponse(status_code=400, content=_envelope("validation_failed", {"errors": exc.errors()}))

    @app.exception_handler(Exception)
    async def fallback(_, exc: Exception):
        return JSONResponse(status_code=500, content=_envelope("internal", {"type": exc.__class__.__name__}))
