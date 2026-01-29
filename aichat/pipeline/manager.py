import logging
from typing import Dict, Tuple

from fastapi import WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription

from aichat.db_models.chat import Chat
from aichat.db_models.db import Session
from aichat.pipeline.processor import Processor
from aichat.pipeline.factory import ModelFactory
from aichat.pipeline.memory import Memory
from aichat.types import MESSAGE_TYPE_AVATAR_INITIALIZE

INPUT_ANALYZER_AUDIO = "zipformer"
INPUT_ANALYZER_VIDEO = "deepface"

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage RTC connection lifecycle, SDP exchanges and binding core processor with connection."""

    _conns: Dict[int, Tuple[RTCPeerConnection, Processor]] = {}

    def __init__(self): ...

    async def register(
        self, chat: Chat, sdp: str, ws: WebSocket, db: Session
    ) -> RTCSessionDescription:
        # proc = Processor(
        #     speech=INPUT_ANALYZER_AUDIO,
        #     video=INPUT_ANALYZER_VIDEO,
        #     llm=chat.llm,
        #     voice=chat.voice,
        #     face=chat.face,
        # )
        # Create a memory
        mem = Memory(chat=chat, db=db, ws=ws)
        # Create a processor
        proc = Processor(
            speech="dummy",
            video="dummy",
            llm="dummy",
            tts="dummy",
            voice=chat.voice,
            memory=mem,
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
        self._conns[chat.id] = rtc, proc  # type: ignore

        # Bind processor with its I/O
        await proc.bind(rtc_in=rtc, ws_out=ws)

        # Initiate SDP Exchange process
        await rtc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await rtc.createAnswer()
        await rtc.setLocalDescription(answer)

        return rtc.localDescription

    async def deregister(self, id: int):
        assert id in self._conns, "connection id not found."

        rtc, proc = self._conns[id]

        await rtc.close()

        await proc.close()

        del self._conns[id]
