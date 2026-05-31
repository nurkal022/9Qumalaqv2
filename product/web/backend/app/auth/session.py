from dataclasses import dataclass
from app.db.models import User, AnonSession


@dataclass
class CurrentSession:
    user: User | None
    anon: AnonSession | None

    @property
    def kind(self) -> str:
        return "user" if self.user else "anon"

    @property
    def owner_filter(self) -> dict:
        return {"user_id": self.user.id} if self.user else {"anon_session_id": self.anon.id}
