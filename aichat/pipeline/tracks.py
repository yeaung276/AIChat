import asyncio
from aiortc import VideoStreamTrack, AudioStreamTrack, MediaStreamTrack
from av import AudioFrame


class AudioOutTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        
    async def add_frame(self, frame):
        await self.queue.put(frame)

    async def recv(self):
        try:
            frame = self.queue.get_nowait()
            # (call once to advance timestamp & maintain sync)
            _ = await super().recv()
        except asyncio.QueueEmpty:
            # fallback: dummy silence with built-in pacing
            frame = await super().recv()

        return frame
    
  
class VideoOutTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        
    async def add_frame(self, frame):
        await self.queue.put(frame)

    async def recv(self):
        try:
            frame = self.queue.get_nowait()
            self._last_frame = frame
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
        except asyncio.QueueEmpty:
            frame = await super().recv()
        return frame