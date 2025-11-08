from fastapi import APIRouter, HTTPException, Depends
from db.database import supabase
from schemas.user_schema import UserBase
from services.auth import generate_token, verify_password, get_password_hash, validate_token
from typing import Annotated
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter()
TABLE_NAME = "user"

@router.post("/auth")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    existing_user = supabase.table(TABLE_NAME).select("*").eq("username", form_data.username).execute().data
    if existing_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    current_user = existing_user[0]
    if not verify_password(form_data.password, current_user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect password")
    access_token = generate_token(current_user["username"])
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
    
@router.post("/signup",  dependencies=[Depends(validate_token)])
def signup(new_user: UserBase):
    existing_user = supabase.table(TABLE_NAME).select("*").eq("username", new_user.username).execute().data
    if existing_user:
        raise HTTPException(status_code=409, detail="User already existed")
    new_user.password = get_password_hash(new_user.password)
    result = supabase.table(TABLE_NAME).insert(new_user.model_dump()).execute()
    return result.data