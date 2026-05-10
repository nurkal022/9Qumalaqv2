from pydantic import BaseModel


class GameSummary(BaseModel):
    id: int
    mode: str
    opponentLabel: str
    result: str | None
    finalScore: str | None
    side: int
    moveCount: int
    startedAt: str
    finishedAt: str | None
    durationMs: int | None
