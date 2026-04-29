from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class UserBase(BaseModel):
    username: str
    email: str | None = Field(None)


class UserLogin(BaseModel):
    login_id: str
    password: str


class UserSignup(UserBase):
    password: str


class UserInfo(UserBase):
    id: UUID
    role: str
    avatar_url: str | None = Field(None)
    model_config = ConfigDict(from_attributes=True)


class UserInfoUpdate(UserBase):
    role: str


class UserPasswordUpdate(BaseModel):
    password: str


class UserAvatarUpdate(BaseModel):
    avatar: str
