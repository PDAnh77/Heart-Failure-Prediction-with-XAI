from fastapi import APIRouter, Request, Response
from schemas.user_schema import UserBase, UserLogin, UserSignup
from services.user_service import (
    login_user,
    google_login,
    google_callback,
    refresh_access_token,
    create_user,
    logout_user
)

router = APIRouter()

@router.get("/google")
async def auth_google(request: Request):
    return await google_login(request)

@router.get("/google/callback")
async def auth_google_callback(request: Request):
    return await google_callback(request)

@router.post("/login")
def login(request: Request, response: Response, user_login: UserLogin):
    return login_user(request, response, user_login)

@router.post("/refresh")
def refresh(request: Request, response: Response):
    return refresh_access_token(request, response)

@router.post("/signup", response_model=UserBase)
def signup(new_user: UserSignup):
    return create_user(new_user)

@router.post("/logout")
def logout(request: Request, response: Response):
    return logout_user(request, response)