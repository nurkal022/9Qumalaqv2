from pydantic import BaseModel, Field
from typing import Literal


class ClockOut(BaseModel):
    initialMs: int
    incrementMs: int
    whiteMs: int
    blackMs: int
    runningSide: Literal[0, 1] | None


class MoveOut(BaseModel):
    ply: int
    side: int
    actor: str
    moveUci: str
    fenAfter: str
    evalCp: int | None = None
    evalDepth: int | None = None
    thinkTimeMs: int | None = None
    clockAfterMs: int | None = None


class EventOut(BaseModel):
    plyAt: int
    actor: str
    type: str
    payload: dict | None = None


class GameStateOut(BaseModel):
    id: int
    mode: str
    side: int
    status: str
    result: str | None = None
    resultReason: str | None = None
    finalScore: str | None = None
    startFen: str
    currentFen: str
    currentPly: int
    sideToMove: int
    clock: ClockOut
    engineThinking: bool
    hintsUsed: int
    hintsLimit: int
    moves: list[MoveOut] = []
    events: list[EventOut] = []
    startedAt: str
    finishedAt: str | None = None


class NewGameReq(BaseModel):
    side: int = Field(ge=0, le=1)
    engineLevel: Literal["easy", "normal", "hard"] = "normal"
    clock: dict | None = None  # {"initialMs": int, "incrementMs": int}
    useBook: bool = False
    startFen: str | None = None


class MoveReq(BaseModel):
    moveUci: str = Field(min_length=2, max_length=8)


class TakebackReq(BaseModel):
    toPly: int = Field(ge=0)
