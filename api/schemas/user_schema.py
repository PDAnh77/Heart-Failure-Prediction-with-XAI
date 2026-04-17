from uuid import UUID
from pydantic import BaseModel, Field


class UserBase(BaseModel):
    username: str
    email: str | None = Field(None)


class UserLogin(UserBase):
    password: str


class UserSignup(UserBase):
    password: str


class UserInfo(UserBase):
    id: UUID
    role: str


class UserInfoUpdate(UserBase):
    role: str


class UserPasswordUpdate(BaseModel):
    password: str


class UserAvatarUpdate(BaseModel):
    avatar: str
