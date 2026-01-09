import hashlib
import secrets
from authlib.integrations.starlette_client import OAuth
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from pwdlib import PasswordHash
from core.config import settings

algorithm = "HS256"
secret = settings.SECRET_KEY
password_hash = PasswordHash.recommended()
security_scheme = HTTPBearer()

oauth = OAuth()
oauth.register(
    name="auth_google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    authorize_params={"scope": "openid email profile"},
    access_token_url="https://accounts.google.com/o/oauth2/token",
    jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
    client_kwargs={"scope": "openid profile email"},
)

def verify_password(plain_password: str, hashed_password: str):
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return password_hash.hash(password)

def generate_access_token(user_id: str):
    expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "access"
    }
    encode_jwt = jwt.encode(payload, secret, algorithm=algorithm)
    return encode_jwt

def generate_refresh_token():
    return secrets.token_urlsafe(64)

def hash_token(token: str):
    return hashlib.sha256(token.encode()).hexdigest()

def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload["sub"]

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")