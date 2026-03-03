import json
import uuid
import logging
from typing import List

from sqlmodel import select
from fastapi import APIRouter, WebSocket, Depends
from fastapi.exceptions import HTTPException

from aichat.db_models.chat import Chat, Feedback
from aichat.schemas.chat import ChatRequest, FeedbackRequest
from aichat.security.auth import get_current_user, get_current_user_ws
from aichat.db_models.db import get_session, Session
from aichat.pipeline.manager import ConnectionManager
from aichat.types import MESSAGE_TYPE_SDP_ANSWER, MESSAGE_TYPE_SDP_OFFER


router = APIRouter(prefix="/api")
conn_mg = ConnectionManager()


@router.websocket("/ws")
async def sdp_exchange(
    ws: WebSocket, user=Depends(get_current_user_ws), db: Session = Depends(get_session)
):
    await ws.accept()

    try:
        async for message in ws.iter_text():
            data = json.loads(message)

            if data.get("type") == MESSAGE_TYPE_SDP_OFFER:
                if not data.get("chat_id"):
                    await ws.send_json(
                        {"type": "error", "message": "chat_id is required."}
                    )
                    continue

                chat = db.exec(
                    select(Chat)
                    .where(Chat.id == data.get("chat_id"))
                    .where(Chat.user_id == user.id)
                ).first()
                if not chat:
                    await ws.send_json({"type": "error", "message": "chat not found."})
                    continue

                logging.info("accepting sdp offer and initializing chat ...")

                answer = await conn_mg.register(chat, data["sdp"], ws=ws)
                await ws.send_json(
                    {
                        "type": MESSAGE_TYPE_SDP_ANSWER,
                        "sdp": answer.sdp,
                    }
                )
                continue
            else:
                logging.warning("Unrecognized message type: %s", data.get("type"))
                await ws.send_json(
                    {"type": "error", "message": "Unrecognized message type."}
                )

    except Exception as e:
        logging.error(e)


@router.post("/chat")
async def create_chat(
    req: ChatRequest, user=Depends(get_current_user), db=Depends(get_session)
) -> Chat:
    chat = Chat(
        name=req.name,
        voice=req.agent.voice,
        face=req.agent.face,
        prompt=req.agent.prompt,
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

@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_session)):
    feedback = db.get(Feedback, req.session_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail="Feedback session not found.")

    feedback.q1_rating = req.q1
    feedback.q2_rating = req.q2
    feedback.q3_rating = req.q3
    feedback.q4_rating = req.q4
    feedback.q5_answer = req.q5

    db.commit()