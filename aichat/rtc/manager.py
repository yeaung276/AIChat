from typing import Dict, Tuple
import uuid

from fastapi import WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription

from aichat.pipeline.processor import Processor

class ConnectionManager:
    def __init__(self):
        self._conns: Dict[uuid.UUID, Tuple[RTCPeerConnection, WebSocket, Processor]] = {}
        
    async def register(self, id: uuid.UUID, ws: WebSocket):
        rtc = RTCPeerConnection()
        proc = Processor()
        
        self._conns[id] = rtc, ws, proc
        
        @rtc.on("connectionstatechange")
        async def on_state_change():
            state = rtc.connectionState

            if state in ("failed", "closed", "disconnected"):
                # Prevent recursive call if already removed
                if id in self._conns:
                    await self.remove_rtc(id)
        
        return id
    
    async def remove_rtc(self, id: uuid.UUID):
        rtc, _, _ = self._conns[id]
        assert rtc is not None, "connection id not found."
        await rtc.close()
        del self._conns[id]
    
    async def accept_offer(self, id: uuid.UUID, sdp: str) -> str:
        rtc, ws, proc = self._conns[id]
        assert rtc is not None, "connection id not found."
        await rtc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type="offer")
        )
        
        answer = await rtc.createAnswer()
        await rtc.setLocalDescription(answer)
        
        proc.bind(rtc, ws)
        
        return rtc.localDescription.sdp