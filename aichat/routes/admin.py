import os
from typing import List

from sqlmodel import select
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from aichat.db_models.chat import ChatSession
from aichat.db_models.db import get_session, Session


_api_key_header = APIKeyHeader(name="X-API-Key")


def verify_admin_api_key(api_key: str = Security(_api_key_header)):
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Admin API key not configured.")
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid admin API key.")


router = APIRouter(prefix="/api/admin", dependencies=[Depends(verify_admin_api_key)])


@router.get("/sessions")
async def get_all_sessions(db: Session = Depends(get_session)) -> List[ChatSession]:
    return db.exec(select(ChatSession)).all() # type: ignore
