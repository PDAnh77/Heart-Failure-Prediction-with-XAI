from fastapi import APIRouter, HTTPException, Depends, Response
from db.database import supabase
from schemas.user_schema import UserBase
from services.auth_service import generate_token, verify_password, get_password_hash, validate_token

router = APIRouter()
TABLE_NAME = "user"

@router.post("/auth")
def login(response: Response, data: UserBase):
    existing_user = supabase.table(TABLE_NAME).select("*").eq("username", data.username).execute().data
    if not existing_user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    current_user = existing_user[0]
    if not verify_password(data.password, current_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = generate_token(current_user["username"])
    
    response.set_cookie(
        key="session",
        value=access_token,
        secure=True,
        httponly=True,
        samesite="lax",
        max_age=60*60,
        path="/"
    )
    return {"username": current_user["username"]}

@router.get("/me")
def get_me(username: str = Depends(validate_token)):
    return {"username": username}
    
@router.post("/signup",  dependencies=[Depends(validate_token)])
def signup(new_user: UserBase):
    existing_user = supabase.table(TABLE_NAME).select("*").eq("username", new_user.username).execute().data
    if existing_user:
        raise HTTPException(status_code=409, detail="User already existed")
    new_user.password = get_password_hash(new_user.password)
    result = supabase.table(TABLE_NAME).insert(new_user.model_dump()).execute()
    return result.data

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="session", path="/")
    return {"detail": "Logged out"}