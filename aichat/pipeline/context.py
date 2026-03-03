from typing import Literal
from fastapi import WebSocket

from sqlmodel import Session, update

from aichat.db_models.chat import Chat
from aichat.db_models.db import engine
from aichat.types import MESSAGE_TYPE_TRANSCRIPT
from aichat.utils.prompt import build_prompt


class Context:
    def __init__(self, chat: Chat, ws: WebSocket):
        self.chat_id = chat.id
        self.prompt = chat.prompt
        self.messages = chat.transcripts
        self.ws = ws

    async def add(self, actor: Literal["user", "assistant"], message: str):
        self.messages.append({"actor": actor, "message": message})
        await self.ws.send_json(
            {
                "type": MESSAGE_TYPE_TRANSCRIPT,
                "data": {"actor": actor, "message": message},
            }
        )
        with Session(engine) as session:
            session.exec(
                update(Chat)
                .where(Chat.id == self.chat_id)  # type: ignore[arg-type]
                .values(transcripts=self.messages)
            )
            session.commit()

    async def get_context(self, emotion: str):
        return build_prompt(
            situation=self.prompt, emotion=emotion, user=self.messages[-1]["message"]
        )