import json
import uuid
import logging
from typing import List

from sqlmodel import select
from fastapi import APIRouter, WebSocket, Depends
from fastapi.exceptions import HTTPException

from aichat.db_models.chat import Chat
from aichat.schemas.chat import ChatRequest
from aichat.security.auth import get_current_user
from aichat.db_models.db import get_session, Session
from aichat.pipeline.manager import ConnectionManager
from aichat.types import MESSAGE_TYPE_SDP_ANSWER, MESSAGE_TYPE_SDP_OFFER


router = APIRouter()
conn_mg = ConnectionManager()


@router.websocket("/ws")
async def sdp_exchange(
    ws: WebSocket, user=Depends(get_current_user), db:Session = Depends(get_session)
):
    await ws.accept()

    try:
        async for message in ws.iter_text():
            data = json.loads(message)

            if data.get("type") == MESSAGE_TYPE_SDP_OFFER:
                if not data.get("chat_id"):
                    await ws.send_json({
                        "type": "error",
                        "message": "chat_id is required."
                    })
                    continue
                
                chat = db.exec(select(Chat).where(Chat.id == data.get("chat_id")).where(Chat.user_id == user.id)).first()
                if not chat:
                    await ws.send_json({
                        "type": "error",
                        "message": "chat not found."
                    })
                    continue
                
                logging.info("accepting sdp offer and initializing chat ...")

                answer = await conn_mg.register(chat, ws, data["sdp"])
                await ws.send_json(
                    {
                        "type": MESSAGE_TYPE_SDP_ANSWER,
                        "sdp": answer.sdp,
                    }
                )
                continue
            else:
                logging.warning("Unrecognized message type: %s", data.get("type"))
                await ws.send_json({
                    "type": "error",
                    "message": "Unrecognized message type."
                })
                

    except Exception as e:
        logging.error(e)


@router.post("/chat")
async def create_chat(
    req: ChatRequest, user=Depends(get_current_user), db=Depends(get_session)
) -> Chat:
    chat = Chat(
        voice=req.agent.voice,
        face=req.agent.face,
        prompt=req.agent.prompt,
        llm=req.dialogue,
        user_id=user.id,
    )

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return chat


@router.get("/chats")
async def get_chats(
    user=Depends(get_current_user), db=Depends(get_session)
) -> List[Chat]:
    return db.exec(select(Chat).where(Chat.user_id == user.id)).all()


@router.get("/chat/{id}")
async def get_transcript(
    id: int, user=Depends(get_current_user), db=Depends(get_session)
) -> Chat:
    chat = db.exec(
        select(Chat).where(Chat.user_id == user.id).where(Chat.id == id)
    ).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return chat
