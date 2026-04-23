from fastapi import APIRouter, Depends, UploadFile, File
from dependencies import get_db
from schemas.user_schema import UserInfo, UserInfoUpdate, UserPasswordUpdate
from services.user_service import get_user_by_id, update_user_by_id, delete_user_by_id, update_user_password, update_user_avatar
from services.auth_service import require_roles
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/me", response_model=UserInfo)
def get_me(user=Depends(require_roles(["admin", "user"])), db: Session = Depends(get_db)):
    return get_user_by_id(db, user["user_id"])


@router.get("/{user_id}", dependencies=[Depends(require_roles(["admin"]))], response_model=UserInfo)
def get_user(user_id: str, db: Session = Depends(get_db)):
    return get_user_by_id(db, user_id)


@router.put("/me/password")
def update_password(
    password_update: UserPasswordUpdate,
    user=Depends(require_roles(["admin", "user"])),
    db: Session = Depends(get_db),
):
    return update_user_password(db, user["user_id"], password_update.password)


@router.post("/me/avatar", response_model=UserInfo)
def upload_avatar(
    file: UploadFile = File(...),
    user=Depends(require_roles(["admin", "user"])),
    db: Session = Depends(get_db),
):
    return update_user_avatar(db, user["user_id"], file)


@router.put("/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def update_user(user_id: str, user_update: UserInfoUpdate, db: Session = Depends(get_db)):
    return update_user_by_id(db, user_id, user_update)


@router.delete("/{user_id}", dependencies=[Depends(require_roles(["admin"]))])
def delete_user(user_id: str, db: Session = Depends(get_db)):
    return delete_user_by_id(db, user_id)
