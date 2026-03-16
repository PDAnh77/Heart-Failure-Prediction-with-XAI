from datetime import datetime
from uuid import UUID
from pydantic import BaseModel


class RefreshTokenBase(BaseModel):
    user_id: UUID
    token_hash: str
    expires_at: datetime
    revoked: bool
    revoked_at: datetime
