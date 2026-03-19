import re
import time
import base64
import asyncio
from pydantic import BaseModel
from typing import cast, List, TypedDict, Any

from fastapi import WebSocket
from av.audio.resampler import AudioResampler
from aiortc import (
    VideoStreamTrack,
    AudioStreamTrack,
    MediaStreamTrack,
    RTCPeerConnection,
)

from aichat.pipeline.context import Context
from aichat.pipeline.factory import ModelFactory
from aichat.pipeline.response_controller import LatencyController
from aichat.types import MESSAGE_TYPE_AVATAR_SPEAK, MESSAGE_TYPE_TRANSCRIPT


class ProfiledResult(BaseModel):
    class profile(TypedDict):
        component: str
        time: float

    profiled: List[profile] = []
    incoming: str = ""
    response: str = ""
    voice: Any = None


class Processor:
    """Processor will collect and segment incoming frames, process each, and sync generated frames for output."""

    def __init__(self, voice: str, context: Context):
        # I/O
        self.ws: WebSocket
        self.context = context
        self.llm_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        self.gen_interrupt = asyncio.Event()

        # processors
        self.stt = ModelFactory.get_speech_model()
        self.llm = ModelFactory.get_dialogue_model()
        self.video_analyzer = ModelFactory.get_emotion_model()
        self.tts = ModelFactory.get_voice_model(voice=voice)
        self.controller = ModelFactory.get_length_controller()

        # tasks
        self.video_task: asyncio.Task | None = None
        self.audio_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None
        self.tts_task: asyncio.Task | None = None
        



    async def bind(self, rtc_in: RTCPeerConnection, ws_out: WebSocket):
        @rtc_in.on("track")
        async def on_track(track: MediaStreamTrack):
            if track.kind == "video":
                self.video_task = asyncio.create_task(
                    self._read_video_track(cast(VideoStreamTrack, track))
                )
            if track.kind == "audio":
                self.audio_task = asyncio.create_task(
                    self._read_audio_track(cast(AudioStreamTrack, track))
                )

        self.llm_task = asyncio.create_task(self._read_llm_queue(self.llm_queue))
        self.tts_task = asyncio.create_task(self._read_tts_queue(self.tts_queue))
        self.ws = ws_out

    async def close(self):
        if self.video_task:
            self.video_task.cancel()

        if self.audio_task:
            self.audio_task.cancel()

        if self.llm_task:
            self.llm_task.cancel()

        return {
            "metrics": self.controller.summary,
            "transcript": self.context.messages,
        }

    async def _read_audio_track(self, track: AudioStreamTrack):
        resampler = AudioResampler(
            format="s16", layout="mono", rate=self.stt.sample_rate
        )
        while True:
            frame = resampler.resample(await track.recv())[0]  # type: ignore
            pcm = frame.to_ndarray()
            data = await self.stt.accept(
                pcm.flatten(), sample_rate=self.stt.sample_rate
            )
            
            if self.stt.is_speaking():
                await self._interrupt()
            
            if data is not None:
                # clear interrupt flag previously set
                await self._clear_interrupt()
                
                data = re.sub(r'[^\x00-\x7F]+', '', data).lower()
                
                await self.llm_queue.put(
                    ProfiledResult(
                        incoming=data,
                        profiled=[
                            {"component": "stt_out", "time": time.perf_counter()}
                        ],
                    )
                )
                
                await self.ws.send_json(
                    {
                        "type": MESSAGE_TYPE_TRANSCRIPT,
                        "data": {"actor": "user", "message": data},
                    }
                )

            await asyncio.sleep(0)

    async def _read_video_track(self, track: VideoStreamTrack):
        while True:
            await self.video_analyzer.accept(await track.recv())  # type: ignore

    async def _read_llm_queue(self, queue: asyncio.Queue[ProfiledResult]):
        while True:
            data = await queue.get()
            try:
                await self.context.add(actor="user", message=data.incoming)

                response = ""
                async for resp in self.llm.generate(
                    await self.context.get_context(
                        self.video_analyzer.emotion, self.controller.mode
                    )
                ):
                    if await self._is_interrupted():
                        break
                    response = resp
                    
                if await self._is_interrupted() or not response:
                    continue

                data.response = response
                data.profiled.append(
                    {"component": "llm_out", "time": time.perf_counter()}
                )
                await asyncio.gather(
                    self.tts_queue.put(data),
                    self.context.add(actor="assistant", message=response),
                )
            except Exception as e:
                print("Error in llm task", e)
                raise e
                

    async def _read_tts_queue(self, queue: asyncio.Queue[ProfiledResult]):
        while True:
            data = await queue.get()
            try:
                async for audio, meta in self.tts.synthesize(data.response):
                    if await self._is_interrupted():
                        break
                    
                    await self.ws.send_json(
                        {
                            "type": MESSAGE_TYPE_AVATAR_SPEAK,
                            "data": {
                                "audio": base64.b64encode(audio).decode("utf-8"),
                                "meta": meta,
                                "waterfall": data.profiled,
                            },
                        }
                    )

                data.profiled.append({"component": "tts_out", "time": time.perf_counter()})
                await self.ws.send_json(
                    {
                        "type": MESSAGE_TYPE_TRANSCRIPT,
                        "data": {"actor": "assistant", "message": data.response},
                    }
                )
                self.controller.update(data.profiled)
            except Exception as e:
                print("Error in llm task", e)
                raise e

    async def _interrupt(self):
        # set flag
        self.gen_interrupt.set()
        # drain queues
        for q in (self.llm_queue, self.tts_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
    async def _clear_interrupt(self):
        self.gen_interrupt.clear()
        
    async def _is_interrupted(self):
        return self.gen_interrupt.is_set()