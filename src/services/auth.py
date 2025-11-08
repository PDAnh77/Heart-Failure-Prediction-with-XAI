import jwt, os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Annotated
from fastapi import HTTPException, Depends
from pwdlib import PasswordHash
from fastapi.security import OAuth2PasswordBearer

load_dotenv()
algorithm = "HS256"
secret = os.getenv("SECRET_JWT_KEY")
password_hash = PasswordHash.recommended()

def verify_password(plain_password: str, hashed_password: str):
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return password_hash.hash(password)

def generate_token(username: str):
    expire = datetime.now() + timedelta(hours=1)
    to_encode = {
        "exp": expire, "username": username
    }
    encode_jwt = jwt.encode(to_encode, secret, algorithm=algorithm)
    return encode_jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/user/auth")

def validate_token(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, secret, algorithms=algorithm)
        username = payload.get("username")
        if username is None:
            raise credentials_exception
    except jwt.ExpiredSignatureError:
        raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
    