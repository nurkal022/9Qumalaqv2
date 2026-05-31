from datetime import datetime
from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(128))
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="kk")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)


class AnonSession(Base):
    __tablename__ = "anon_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="kk")


class Game(Base):
    __tablename__ = "games"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    anon_session_id: Mapped[str | None] = mapped_column(ForeignKey("anon_sessions.id", ondelete="SET NULL"))
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[int] = mapped_column(Integer, nullable=False)
    opponent_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    opponent_ref: Mapped[str | None] = mapped_column(String(128))
    engine_level: Mapped[str] = mapped_column(String(16), nullable=False, default="normal")
    clock_initial_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    clock_increment_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    clock_white_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    clock_black_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    last_clock_at: Mapped[datetime | None] = mapped_column(DateTime)
    start_fen: Mapped[str] = mapped_column(Text, nullable=False)
    current_fen: Mapped[str] = mapped_column(Text, nullable=False)
    current_ply: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    side_to_move: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[str | None] = mapped_column(String(16))
    result_reason: Mapped[str | None] = mapped_column(String(32))
    final_score: Mapped[str | None] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)

    moves: Mapped[list["Move"]] = relationship(back_populates="game", cascade="all, delete-orphan", order_by="Move.ply")
    events: Mapped[list["GameEvent"]] = relationship(back_populates="game", cascade="all, delete-orphan", order_by="GameEvent.created_at")

    __table_args__ = (
        CheckConstraint(
            "(user_id IS NOT NULL AND anon_session_id IS NULL) OR "
            "(user_id IS NULL AND anon_session_id IS NOT NULL)",
            name="games_owner_xor",
        ),
        Index("idx_games_user", "user_id", "started_at"),
        Index("idx_games_anon", "anon_session_id", "started_at"),
        Index("idx_games_active", "status"),
    )


class Move(Base):
    __tablename__ = "moves"
    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    ply: Mapped[int] = mapped_column(Integer, nullable=False)
    side: Mapped[int] = mapped_column(Integer, nullable=False)
    actor: Mapped[str] = mapped_column(String(16), nullable=False)
    move_uci: Mapped[str] = mapped_column(String(8), nullable=False)
    fen_after: Mapped[str] = mapped_column(Text, nullable=False)
    eval_cp: Mapped[int | None] = mapped_column(Integer)
    eval_depth: Mapped[int | None] = mapped_column(Integer)
    pv: Mapped[str | None] = mapped_column(Text)
    think_time_ms: Mapped[int | None] = mapped_column(Integer)
    clock_after_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)

    game: Mapped[Game] = relationship(back_populates="moves")
    __table_args__ = (UniqueConstraint("game_id", "ply"), Index("idx_moves_game", "game_id", "ply"))


class GameEvent(Base):
    __tablename__ = "game_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    ply_at: Mapped[int] = mapped_column(Integer, nullable=False)
    actor: Mapped[str] = mapped_column(String(16), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)

    game: Mapped[Game] = relationship(back_populates="events")
    __table_args__ = (Index("idx_events_game", "game_id", "created_at"),)


class Engine(Base):
    __tablename__ = "engines"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    binary_path: Mapped[str] = mapped_column(Text, nullable=False)
    weights_path: Mapped[str | None] = mapped_column(Text)
    build_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), nullable=False)
