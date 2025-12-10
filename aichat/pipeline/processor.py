import asyncio
from typing import cast

from fastapi import WebSocket
from av.audio.resampler import AudioResampler
from aiortc import VideoStreamTrack, AudioStreamTrack, MediaStreamTrack, RTCPeerConnection

from .config import config_resolver
from aichat.types import MESSAGE_TYPE_SPEECH_SPEAK





class Processor:
    """Processor will collect and segment incoming frames, process each, and sync generated frames for output."""

    def __init__(self, rtc: RTCPeerConnection, ws: WebSocket):
        # I/O 
        self.websocket: WebSocket
        self.llm_queue = asyncio.Queue()
        
        # processors
        processors = config_resolver({})
        self.stt = processors["tts_model"]
        self.llm = processors["llm_model"]
        
        # tasks
        self.video_task: asyncio.Task
        self.audio_task: asyncio.Task
        self.llm_task: asyncio.Task
        self.bind(rtc, ws)
        
    
    def bind(self, rtc: RTCPeerConnection, ws: WebSocket):
        self.websocket = ws
        
        @rtc.on("track")
        async def on_track(track: MediaStreamTrack):
            if track.kind == "video":
                self.video_task = asyncio.create_task(self._read_video_track(cast(VideoStreamTrack, track)))
            if track.kind == "audio":
                self.audio_task = asyncio.create_task(self._read_audio_track(cast(AudioStreamTrack, track)))
        
        self.llm_task = asyncio.create_task(self._read_llm_queue(self.llm_queue))
        

    async def _read_audio_track(self, track: AudioStreamTrack):
        resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        while True:
            frame = resampler.resample(await track.recv())[0] # type: ignore
            pcm = frame.to_ndarray() 
            message = self.stt.accept(pcm.flatten())
            if message is not None:
                await self.llm_queue.put(message)
                
            await asyncio.sleep(0)

    async def _read_video_track(self, track: VideoStreamTrack):
        while True:
            frame = await track.recv()
            await asyncio.sleep(0)
            
    async def _read_llm_queue(self, queue: asyncio.Queue):
        while True:
            message = await queue.get()
            final = ""
            async for resp in self.llm.generate(message):
                final = resp
            await self.websocket.send_json({
                "type": MESSAGE_TYPE_SPEECH_SPEAK,
                "data": final
            })
                
                
        