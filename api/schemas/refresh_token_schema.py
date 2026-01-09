from datetime import datetime
from pydantic import BaseModel

class RefreshTokenBase(BaseModel):
    user_id: str
    token_hash: str
    expires_at: datetime
    revoked: bool
    revoked_at: datetime