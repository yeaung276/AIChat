import uuid
from typing import Dict, Tuple

from fastapi import WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription

from aichat.pipeline.processor import Processor
from aichat.db_models.chat import Chat


class ConnectionManager:
    """Manage RTC connection lifecycle, SDP exchanges and binding core processor with connection."""
    _conns: Dict[uuid.UUID, Tuple[RTCPeerConnection, Processor]] = {}

    def __init__(self): ...

    async def register(self, chat: Chat, ws: WebSocket, sdp: str) -> RTCSessionDescription:
        rtc = RTCPeerConnection()
        proc = Processor(
            speech="zipformer",
            video="deepface",
            llm=chat.llm,
            voice=chat.voice,
            face=chat.face,
        )

        @rtc.on("connectionstatechange")
        async def on_state_change():
            state = rtc.connectionState

            if state in ("failed", "closed", "disconnected"):
                if chat.id in self._conns:
                    print("closing connection due to invalid state change: ", state)
                    await self.deregister(chat.id) # type: ignore
                    
        
        await proc.bind(rtc, ws)
        
        await rtc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        
        answer = await rtc.createAnswer()
        await rtc.setLocalDescription(answer)
        
        return rtc.localDescription

    async def deregister(self, id: uuid.UUID):
        rtc, proc = self._conns[id]
        assert rtc is not None, "connection id not found."
        await rtc.close()
        
        assert proc is not None, "connection id not found."
        await proc.close()
        
        del self._conns[id]
