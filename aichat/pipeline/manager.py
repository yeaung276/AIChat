import logging
import asyncio
from typing import Dict, Tuple, cast

from fastapi import WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription

from sqlmodel import Session, update

from aichat.db_models.chat import Character, ChatSession
from aichat.db_models.db import engine
from aichat.pipeline.processor import Processor
from aichat.pipeline.factory import ModelFactory
from aichat.pipeline.context import Context
from aichat.types import MESSAGE_TYPE_AVATAR_INITIALIZE, MESSAGE_TYPE_FEEDBACK_ID


logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage RTC connection lifecycle, SDP exchanges and binding core processor with connection."""

    _conns: Dict[int, Tuple[RTCPeerConnection, Processor, WebSocket]] = {}

    async def register(
        self, char: Character, sdp: str, ws: WebSocket
    ) -> RTCSessionDescription:
        # Create a memory
        mem = Context(prompt=char.prompt, ws=ws)
        # Create a processor
        proc = Processor(voice=char.voice, context=mem)

        # Create RTC object with connection lifecycle
        rtc = RTCPeerConnection()

        @rtc.on("connectionstatechange")
        async def on_state_change():
            state = rtc.connectionState

            if state in ("failed", "closed", "disconnected") and char.id in self._conns:
                logger.warning(
                    "closing connection due to invalid state change: %s", state
                )
                await self.deregister(char.id)  # type: ignore

            elif state in ("connected") and char.id in self._conns:
                logger.warning("rtc connected. Initializing avatar...")
                await ws.send_json(
                    {
                        "type": MESSAGE_TYPE_AVATAR_INITIALIZE,
                        "data": {
                            "avatar": ModelFactory.get_avatar(char.face),
                        },
                    }
                )

        # Register connection
        self._conns[cast(int, char.id)] = rtc, proc, ws

        # Bind processor with its I/O
        await proc.bind(rtc_in=rtc, ws_out=ws)

        # Initiate SDP Exchange process
        await rtc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await rtc.createAnswer()
        await rtc.setLocalDescription(answer)

        return rtc.localDescription

    async def deregister(self, id: int):
        assert id in self._conns, "connection id not found."

        rtc, proc, ws = self._conns[id]

        await rtc.close()

        summary = await proc.close()

        del self._conns[id]

        feedback = self._save_summary(summary)

        await ws.send_json(
            {
                "type": MESSAGE_TYPE_FEEDBACK_ID,
                "data": {
                    "id": feedback.id,
                },
            }
        )

    def _save_summary(self, summary: dict):
        metrics = summary["metrics"]

        feedback = ChatSession(
            mean_latency_ms=metrics["mean_latency_ms"],
            max_latency_ms=metrics["max_latency_ms"],
            min_latency_ms=metrics["min_latency_ms"] if metrics["turns"] else 0.0,
            session_duration_s=metrics["session_duration_s"],
            session_turns=metrics["turns"],
            q1_rating=0,
            q2_rating=0,
            q3_rating=0,
            q4_rating=0,
            q5_answer="",
        )
        with Session(engine) as session:
            session.add(feedback)
            session.commit()
            session.refresh(feedback)

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._save_transcript, summary["transcript"])

        return feedback

    def _save_transcript(self, transcripts: list):
        import spacy

        nlp = spacy.load("en_core_web_sm")

        safe_transcripts = []
        for msg in transcripts:
            doc = nlp(msg["message"])
            result = msg["message"]
            for ent in reversed(doc.ents):
                if ent.label_ in {"PERSON", "GPE", "LOC", "ORG"}:
                    result = (
                        result[: ent.start_char]
                        + f"[{ent.label_}]"
                        + result[ent.end_char :]
                    )
                safe_transcripts.append({"actor": msg["actor"], "message": result})

        with Session(engine) as session:
            session.exec(
                update(ChatSession)
                .where(ChatSession.id == self.chat_id)  # type: ignore
                .values(transcripts=safe_transcripts)
            )
            session.commit()
