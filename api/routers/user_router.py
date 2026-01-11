from fastapi import APIRouter, Depends, Request, Response
from schemas.user_schema import UserBase
from services.user_service import (
    login_user,
    google_login,
    google_callback,
    refresh_access_token,
    get_current_user,
    signup_user,
    logout_user,
)
from services.auth_service import validate_token

router = APIRouter()

@router.post("/auth/login")
def login(request: Request, response: Response, data: UserBase):
    return login_user(request, response, data)

@router.get("/auth/google")
async def auth_google(request: Request):
    return await google_login(request)

@router.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    return await google_callback(request)

@router.post("/auth/refresh")
def refresh(request: Request, response: Response):
    return refresh_access_token(request, response)

@router.get("/user/me")
def me(user_id: str = Depends(validate_token)):
    return get_current_user(user_id)

@router.post("/user/signup", dependencies=[Depends(validate_token)])
def signup(new_user: UserBase):
    return signup_user(new_user)

@router.post("/auth/logout")
def logout(request: Request, response: Response):
    return logout_user(request, response)
