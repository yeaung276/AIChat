import logging
from typing import Dict, Tuple

from fastapi import WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription

from sqlmodel import Session

from aichat.db_models.chat import Chat, Feedback
from aichat.db_models.db import engine
from aichat.pipeline.processor import Processor
from aichat.pipeline.factory import ModelFactory
from aichat.pipeline.context import Context
from aichat.types import MESSAGE_TYPE_AVATAR_INITIALIZE, MESSAGE_TYPE_FEEDBACK_ID

INPUT_ANALYZER_AUDIO = "zipformer"
INPUT_ANALYZER_VIDEO = "deepface"
DIALOGUE_PROCESSOR = "qwen2.5"
OUTPUT_SYNTHESIZER_AUDIO = "kokoro"

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage RTC connection lifecycle, SDP exchanges and binding core processor with connection."""

    _conns: Dict[int, Tuple[RTCPeerConnection, Processor, WebSocket]] = {}

    async def register(self, chat: Chat, sdp: str, ws: WebSocket) -> RTCSessionDescription:
        # Create a memory
        mem = Context(chat=chat, ws=ws)
        # Create a processor
        proc = Processor(
            speech=INPUT_ANALYZER_AUDIO,
            video=INPUT_ANALYZER_VIDEO,
            llm=DIALOGUE_PROCESSOR,
            tts=OUTPUT_SYNTHESIZER_AUDIO,
            voice=chat.voice,
            context=mem,
        )

        # Create RTC object with connection lifecycle
        rtc = RTCPeerConnection()

        @rtc.on("connectionstatechange")
        async def on_state_change():
            state = rtc.connectionState

            if state in ("failed", "closed", "disconnected") and chat.id in self._conns:
                logger.warning(
                    "closing connection due to invalid state change: %s", state
                )
                await self.deregister(chat.id)  # type: ignore

            elif state in ("connected") and chat.id in self._conns:
                logger.warning("rtc connected. Initializing avatar...")
                await ws.send_json(
                    {
                        "type": MESSAGE_TYPE_AVATAR_INITIALIZE,
                        "data": {
                            "avatar": ModelFactory.get_avatar(chat.face),
                        },
                    }
                )

        # Register connection
        self._conns[chat.id] = rtc, proc, ws

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

        metrics = await proc.close()

        del self._conns[id]

        feedback = self._save_metrics(metrics)
        await ws.send_json({
            "type": MESSAGE_TYPE_FEEDBACK_ID,
            "data": {
                "id": feedback.id,
            },
        })

    def _save_metrics(self, metrics: dict):
        feedback = Feedback(
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
        
        return feedback

