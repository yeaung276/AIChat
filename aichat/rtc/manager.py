from typing import Dict
import uuid

from aiortc import RTCPeerConnection, RTCSessionDescription

from aichat.pipeline import Processor

class RTCManager:
    def __init__(self):
        self._conns: Dict[uuid.UUID, RTCPeerConnection] = {}
        
    async def create_rtc(self, id: uuid.UUID):
        rtc = RTCPeerConnection()
        self._conns[id] = rtc
        return id
    
    async def remove_rtc(self, id: uuid.UUID):
        rtc = self._conns[id]
        assert rtc is not None, "connection id not found."
        await rtc.close()
        del self._conns[id]
    
    async def accept_offer(self, id: uuid.UUID, sdp: str) -> str:
        rtc = self._conns[id]
        assert rtc is not None, "connection id not found."
        processor = Processor()
        processor.bind(rtc)
        await rtc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type="offer")
        )
        
        answer = await rtc.createAnswer()
        await rtc.setLocalDescription(answer)
        
        return rtc.localDescription.sdp