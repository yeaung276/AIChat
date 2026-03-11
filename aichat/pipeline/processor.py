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
from aichat.types import MESSAGE_TYPE_AVATAR_SPEAK


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

    def __init__(
        self, speech: str, video: str, llm: str, tts: str, voice: str, context: Context
    ):
        # I/O
        self.ws: WebSocket
        self.context = context
        self.llm_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()

        # processors
        self.stt = ModelFactory.get_speech_model(speech)
        self.llm = ModelFactory.get_dialogue_model(llm)
        self.video_analyzer = ModelFactory.get_emotion_model(video)
        self.tts = ModelFactory.get_voice_model(tts, voice=voice)

        # tasks
        self.video_task: asyncio.Task | None = None
        self.audio_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None
        self.tts_task: asyncio.Task | None = None
        
        # metrics
        self.metrics = {
            "session_start": 0.0,
            "latency_sum": 0.0,
            "max_latency": 0.0,
            "min_latency": float("inf"),
            "session_turns": 0,
        }

    def start_session(self):
        if not self.metrics["session_start"]:
            self.metrics["session_start"] = time.perf_counter()

    def update_metrics(self, profiled: list):
        stt_time = next(p["time"] for p in profiled if p["component"] == "stt_out")
        tts_time = next(p["time"] for p in profiled if p["component"] == "tts_out")
        latency_ms = (tts_time - stt_time) * 1000
        self.metrics["latency_sum"] += latency_ms
        self.metrics["max_latency"] = max(self.metrics["max_latency"], latency_ms)
        self.metrics["min_latency"] = min(self.metrics["min_latency"], latency_ms)
        self.metrics["session_turns"] += 1

    async def bind(self, rtc_in: RTCPeerConnection, ws_out: WebSocket):
        @rtc_in.on("track")
        async def on_track(track: MediaStreamTrack):
            self.start_session()
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
            
        turns = self.metrics["session_turns"]
        return {
            "mean_latency_ms": self.metrics["latency_sum"] / turns if turns else 0.0,
            "max_latency_ms": self.metrics["max_latency"],
            "min_latency_ms": self.metrics["min_latency"],
            "turns": turns,
            "session_duration_s": time.perf_counter() - self.metrics["session_start"]
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
            if data is not None:
                await self.llm_queue.put(
                    ProfiledResult(
                        incoming=data.lower(),
                        profiled=[{"component": "stt_out", "time": time.perf_counter()}],
                    )
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
                    await self.context.get_context(self.video_analyzer.emotion)
                ):
                    response = resp

                data.response = response
                data.profiled.append({"component": "llm_out", "time": time.perf_counter()})
                t1 = self.tts_queue.put(data)
                t2 = self.context.add(actor="assistant", message=response)
                await asyncio.gather(t1, t2)
            except Exception as e:
                continue

    async def _read_tts_queue(self, queue: asyncio.Queue[ProfiledResult]):
        while True:
            data = await queue.get()

            async for audio, meta in self.tts.synthesize(data.response):
                data.profiled.append({"component": "tts_out", "time": time.perf_counter()})
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
            self.update_metrics(data.profiled)
