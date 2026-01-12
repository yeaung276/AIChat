from sqlmodel import select
from fastapi import APIRouter, Depends, Response, HTTPException

from aichat.schemas.user import LoginRequest, RegisterRequest
from aichat.db_models.db import get_session, Session
from aichat.db_models.user import User
from aichat.security.auth import (
    verify_password,
    create_session_token,
    hash_password,
    get_current_user,
    SESSION_COOKIE_NAME,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)


router = APIRouter()


@router.post("/login")
async def login(
    req: LoginRequest, res: Response, sess: Session = Depends(get_session)
) -> User:
    user = sess.exec(select(User).where(User.username == req.username)).first()

    if not user or not verify_password(req.password, user.pwd_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_session_token(user.id)  # type: ignore

    res.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    return user


@router.post("/register")
async def register(
    req: RegisterRequest, res: Response, db: Session = Depends(get_session)
) -> User:
    from sqlalchemy.exc import IntegrityError

    # Validate username is not empty
    if not req.username or not req.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")

    user = User(
        username=req.username,
        pwd_hash=hash_password(req.password),
        screen_name=req.name,
        bio=req.bio,
    )

    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username already exists")

    token = create_session_token(user.id)  # type: ignore

    res.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    return user

@router.get("/me")
async def me(user = Depends(get_current_user)) -> User:
    return user
