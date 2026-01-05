from pydantic import BaseModel, Field

class UserBase(BaseModel):
    username: str
    email: str | None = Field(None)
    password: str

class UserLogin(UserBase):
    id: str
    
class UserGet(BaseModel):
    id: str
    username: str
    email: str