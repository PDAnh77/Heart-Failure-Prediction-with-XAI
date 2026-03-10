from pydantic import BaseModel, Field

class UserBase(BaseModel):
    username: str
    email: str | None = Field(None)

class UserLogin(UserBase):
    password: str

class UserSignup(UserBase):
    password: str
    
class UserInfo(UserBase):
    id: str
    role: str

class UserUpdate(UserBase):
    role: str