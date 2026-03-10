from fastapi import APIRouter, Depends
from schemas.user_schema import UserUpdate
from services.user_service import (get_user_by_id, update_user_by_id, delete_user_by_id)
from services.auth_service import require_roles

router = APIRouter()

@router.get("/me")
def get_me(user = Depends(require_roles(["admin", "viewer"]))):
    return get_user_by_id(user["user_id"])

@router.get("/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def get_user(user_id: str):
    return get_user_by_id(user_id)

@router.put("/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def update_user(user_id: str, user_update: UserUpdate):
    return update_user_by_id(user_id, user_update)

@router.delete("/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def delete_user(user_id: str):
    return delete_user_by_id(user_id)