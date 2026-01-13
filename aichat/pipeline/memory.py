from typing import Literal
from fastapi import WebSocket

from sqlmodel import update

from aichat.db_models.chat import Chat
from aichat.db_models.db import Session
from aichat.types import MESSAGE_TYPE_TRANSCRIPT


class Memory:
    def __init__(self, chat: Chat, db: Session, ws: WebSocket):
        self.chat_id = chat.id
        self.prompt = chat.prompt
        self.messages = chat.transcripts
        self.ws = ws
        self.db = db
        self.chat = chat

    async def add(self, actor: Literal["user", "assistant"], message: str):
        self.messages.append({"actor": actor, "message": message})
        await self.ws.send_json(
            {
                "type": MESSAGE_TYPE_TRANSCRIPT,
                "data": {"actor": actor, "message": message},
            }
        )
        self.db.exec(
            update(Chat)
            .where(Chat.id == self.chat_id) # type: ignore
            .values(transcripts=self.messages)
        )

    async def get_context(self):
        return self.messages[-1]["message"]
