from pydantic import BaseModel, Field, field_validator


class RegisterReq(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=6, max_length=128)
    locale: str = Field(default="kk", pattern="^(kk|ru)$")

    @field_validator("username")
    @classmethod
    def lowercase_alnum(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("invalid characters")
        return v


class LoginReq(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    display_name: str | None = Field(default=None, serialization_alias="displayName")
    locale: str

    model_config = {"from_attributes": True}


class MeOut(BaseModel):
    kind: str  # 'user' | 'anon'
    user: UserOut | None = None
    anonId: str | None = None
