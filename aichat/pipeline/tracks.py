import asyncio
from aiortc import VideoStreamTrack, AudioStreamTrack, MediaStreamTrack
from av import AudioFrame

MIN_BUFFER_FRAME = 5

class AudioOutTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()
        self.sampling_rate = 24_000

    async def recv(self):
        try:
            if self.queue.qsize() < MIN_BUFFER_FRAME:
                raise asyncio.QueueEmpty()
            frame = self.queue.get_nowait()
            # (call once to advance timestamp & maintain sync)
            frame.sample_ra
        except asyncio.QueueEmpty:
            # fallback: dummy silence with built-in pacing
            frame = await super().recv()

        return frame
    
  
class VideoOutTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()

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