import asyncio
from typing import cast

from fastapi import WebSocket
from av.audio.resampler import AudioResampler
from aiortc import (
    VideoStreamTrack,
    AudioStreamTrack,
    MediaStreamTrack,
    RTCPeerConnection,
)

from aichat.pipeline.memory import Memory
from aichat.pipeline.factory import ModelFactory
from aichat.types import MESSAGE_TYPE_AVATAR_SPEAK, MESSAGE_TYPE_TRANSCRIPT


class Processor:
    """Processor will collect and segment incoming frames, process each, and sync generated frames for output."""

    def __init__(self, speech: str, video: str, llm: str, voice: str, memory: Memory):
        # I/O
        self.ws: WebSocket
        self.mem = memory
        self.llm_queue = asyncio.Queue()

        # processors
        self.stt = ModelFactory.get_speech_model(speech)
        self.llm = ModelFactory.get_dialogue_model(llm)
        self.video_analyzer = ModelFactory.get_video_model(video)

        # tasks
        self.video_task: asyncio.Task | None = None
        self.audio_task: asyncio.Task | None = None
        self.llm_task: asyncio.Task | None = None

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
        self.ws = ws_out

    async def close(self):
        if self.video_task:
            self.video_task.cancel()

        if self.audio_task:
            self.audio_task.cancel()

        if self.llm_task:
            self.llm_task.cancel()

    async def _read_audio_track(self, track: AudioStreamTrack):
        resampler = AudioResampler(
            format="s16", layout="mono", rate=self.stt.sample_rate
        )
        while True:
            frame = resampler.resample(await track.recv())[0]  # type: ignore
            pcm = frame.to_ndarray()
            message = await self.stt.accept(
                pcm.flatten(), sample_rate=self.stt.sample_rate
            )
            if message is not None:
                await self.llm_queue.put(message)

            await asyncio.sleep(0)

    async def _read_video_track(self, track: VideoStreamTrack):
        while True:
            await self.video_analyzer.accept(await track.recv())  # type: ignore

    async def _read_llm_queue(self, queue: asyncio.Queue):
        while True:
            message = await queue.get()
            try:
                await self.mem.add(actor="user", message=message)

                response = ""
                async for resp in self.llm.generate(
                    await self.mem.get_context(self.video_analyzer.emotion)
                ):
                    response = resp

                t1 = self.ws.send_json(
                    {"type": MESSAGE_TYPE_AVATAR_SPEAK, "data": {"text": response}}
                )
                t2 = self.mem.add(actor="assistant", message=response)
                await asyncio.gather(t1, t2)
            except Exception as e:
                continue
