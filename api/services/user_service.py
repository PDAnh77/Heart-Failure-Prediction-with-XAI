import random, string
from uuid import UUID
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Request, Response, status
from starlette.responses import RedirectResponse
from db.database import supabase
from core.config import settings
from schemas.user_schema import UserBase, UserInfo, UserLogin, UserSignup, UserUpdate
from schemas.refresh_token_schema import RefreshTokenBase
from services.auth_service import (
    generate_access_token,
    verify_password,
    get_password_hash,
    generate_refresh_token,
    hash_token,
    oauth,
)

TABLE_USER = "users"
TABLE_TOKEN = "refresh_tokens"
REFRESH_TOKEN_TTL_DAYS = 7

def check_uuid(id: str):
    try:
        validated_uuid = str(UUID(id))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid UUID format")
    return validated_uuid

def store_refresh_token(request: Request, user_id: str, refresh_token: str):
    user_agent = request.headers.get("user-agent", "unknown")
    forwarded = request.headers.get("x-forwarded-for")
    ip_address = forwarded.split(",")[0] if forwarded else request.client.host

    supabase.table(TABLE_TOKEN).insert({
        "user_id": user_id,
        "token_hash": hash_token(refresh_token),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_TTL_DAYS)).isoformat(),
        "ip_address": ip_address,
        "user_agent": user_agent
    }).execute()

def set_refresh_token_cookie(response: Response, refresh_token: str):
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        secure=True,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/api/auth"
    )

def login_user(request: Request, response: Response, user_login: UserLogin):
    existing_user = supabase.table(TABLE_USER).select("*").eq("username", user_login.username).execute()

    if not existing_user.data:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    current_user = existing_user.data[0]

    if not verify_password(user_login.password, current_user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    access_token = generate_access_token(current_user["id"], current_user["role"])
    refresh_token = generate_refresh_token()
    store_refresh_token(request, current_user["id"], refresh_token)
    set_refresh_token_cookie(response, refresh_token)

    return {
        "username": current_user["username"],
        "access_token": access_token,
        "token_type": "bearer"
    }

async def google_login(request: Request):
    return await oauth.auth_google.authorize_redirect(
        request,
        settings.REDIRECT_URI,
        prompt="select_account"
    )

async def google_callback(request: Request):
    token = await oauth.auth_google.authorize_access_token(request)
    userinfo = token.get("userinfo")
    email = userinfo.get("email")

    columns = ",".join(UserInfo.model_fields.keys())
    existing_user = supabase.table(TABLE_USER).select(columns).eq("email", email).execute()

    if not existing_user.data:
        prefix = email.split("@")[0].rstrip("0123456789")
        random_digits = "".join(random.choices(string.digits, k=5))
        username = f"{prefix}{random_digits}"
        new_user = supabase.table(TABLE_USER).insert({
            "username": username,
            "email": email
        }).execute()
        user_id = new_user.data[0]["id"]
    else:
        user_id = existing_user.data[0]["id"]

    refresh_token = generate_refresh_token()
    store_refresh_token(request, user_id, refresh_token)
    response = RedirectResponse(url=f"{settings.CLIENT_URL}/predict")
    set_refresh_token_cookie(response, refresh_token)
    return response

def refresh_access_token(request: Request, response: Response):
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token missing")

    token_hash = hash_token(refresh_token)
    columns = ",".join(RefreshTokenBase.model_fields.keys())
    select_query = f"{columns}, users(role)"
    result = supabase.table(TABLE_TOKEN).select(select_query).eq("token_hash", token_hash).execute()
    db_token = result.data[0] if result.data else None

    if not db_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
        
    if db_token.get("revoked"):
        time_elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(db_token.get("revoked_at"))

        if time_elapsed > timedelta(seconds=20):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token reuse detected. Please login again.")
        else:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Refresh in progress, please retry.")
        
    if datetime.fromisoformat(db_token["expires_at"]) < datetime.now(timezone.utc):
        supabase.table(TABLE_TOKEN).update({
            "revoked": True,
            "revoked_at": datetime.now(timezone.utc).isoformat()
        }).eq("token_hash", token_hash).execute()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Expired refresh token")
    
    supabase.table(TABLE_TOKEN).update({
        "revoked": True,
        "revoked_at": datetime.now(timezone.utc).isoformat()
        }).eq("token_hash", token_hash).execute()

    user_role = db_token.get("users", {}).get("role")
    if not user_role:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User associated with this token no longer exists")
    
    new_access_token = generate_access_token(db_token["user_id"], user_role)
    new_refresh_token = generate_refresh_token()
    store_refresh_token(request, db_token["user_id"], new_refresh_token)
    set_refresh_token_cookie(response, new_refresh_token)

    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    }

def get_user_by_id(user_id: str):
    user_uuid = check_uuid(user_id)

    columns = ",".join(UserInfo.model_fields.keys())
    res = supabase.table(TABLE_USER).select(columns).eq("id", user_uuid).execute()
    if not res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return res.data[0]

def create_user(new_user: UserSignup):
    columns = ",".join(UserBase.model_fields.keys())
    user = supabase.table(TABLE_USER).select(columns).eq("username", new_user.username).execute()
    if user.data:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already existed")
    new_user.password = get_password_hash(new_user.password)
    res = supabase.table(TABLE_USER).insert(new_user.model_dump()).execute()
    return res.data[0]

def logout_user(request: Request, response: Response):
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        token_hash = hash_token(refresh_token)
        supabase.table(TABLE_TOKEN).update({
            "revoked": True,
            "revoked_at": datetime.now(timezone.utc).isoformat()
            }).eq("token_hash", token_hash).execute()
    response.delete_cookie(key="refresh_token", path="/api/auth")
    return {"detail": "Logged out successfully"}

def update_user_by_id(user_id: str, user_update: UserUpdate):
    user_uuid = check_uuid(user_id)

    columns = ",".join(UserBase.model_fields.keys())
    user = supabase.table(TABLE_USER).select(columns).eq("id", user_uuid).execute()
    if not user.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    update_data = user_update.model_dump(exclude_unset=True)
    res = supabase.table(TABLE_USER).update(update_data).eq("id", user_uuid).execute()
    return res.data[0]

def delete_user_by_id(user_id: str):
    user_uuid = check_uuid(user_id)
    
    columns = ",".join(UserBase.model_fields.keys())
    user = supabase.table(TABLE_USER).select(columns).eq("id", user_uuid).execute()
    if not user.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    supabase.table(TABLE_USER).delete().eq("id", user_uuid).execute()
    return {"detail": "Delete user successfully"}
