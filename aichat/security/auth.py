import os
from datetime import datetime, timedelta

from jose import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException

from aichat.db_models.db import Session, get_session
from aichat.db_models.user import User


SECRET_KEY = os.getenv("JWT_SECRET", "secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_TOKEN_EXPIRY", "60"))
SESSION_COOKIE_NAME = "session"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_session_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "type": "session",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


from fastapi import Request


def get_current_user(
    request: Request,
    db: Session = Depends(get_session),
):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
