import asyncio
from typing import cast

from fastapi import WebSocket
from av.audio.resampler import AudioResampler
from aiortc import VideoStreamTrack, AudioStreamTrack, MediaStreamTrack, RTCPeerConnection

from aichat.types import MESSAGE_TYPE_SPEECH_SPEAK, MESSAGE_TYPE_SPEECH_DEBUG
from aichat.pipeline.factory import ModelFactory





class Processor:
    """Processor will collect and segment incoming frames, process each, and sync generated frames for output."""

    def __init__(self):
        # I/O 
        self.websocket: WebSocket
        self.llm_queue = asyncio.Queue()
        
        # processors
        self.stt, self.llm, self.video_analyzer = ModelFactory.create_models()
        
        # tasks
        self.video_task: asyncio.Task | None = None
        self.audio_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None
        
    
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
        resampler = AudioResampler(format='s16', layout='mono', rate=self.stt.sample_rate)
        while True:
            frame = resampler.resample(await track.recv())[0] # type: ignore
            pcm = frame.to_ndarray() 
            message = await self.stt.accept(pcm.flatten(), sample_rate=self.stt.sample_rate)
            if message is not None:
                await self.llm_queue.put(message)
                
            await asyncio.sleep(0)

    async def _read_video_track(self, track: VideoStreamTrack):
        while True:
            await self.video_analyzer.accept(await track.recv()) # type: ignore
            
    async def _read_llm_queue(self, queue: asyncio.Queue):
        while True:
            message = await queue.get()
            await self.websocket.send_json({
                "type": MESSAGE_TYPE_SPEECH_DEBUG,
                "data":  {
                    "actor": "user",
                    "message": message 
                }
            })
            response = ""
            async for resp in self.llm.generate(message):
                response = resp
            await self.websocket.send_json({
                "type": MESSAGE_TYPE_SPEECH_SPEAK,
                "data": response
            })
                
                
        