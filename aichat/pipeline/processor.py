import asyncio
from typing import cast

from av.audio.resampler import AudioResampler
from aiortc import VideoStreamTrack, AudioStreamTrack, MediaStreamTrack, RTCPeerConnection

from aichat.components.stt.zipformer import ZipformerSTT

from .tracks import AudioOutTrack, VideoOutTrack


class Processor:
    """Processor will collect and segment incoming frames, process each, and sync generated frames for output."""
    def __init__(self):
        # I/O tracks
        self.audio_in: AudioStreamTrack|None = None
        self.video_in: VideoStreamTrack|None = None
        
        self.audio_out = AudioOutTrack()
        self.video_out = VideoOutTrack()
        
        # processors
        self.stt = ZipformerSTT()
    
        # reading tasks
        asyncio.create_task(self._read_audio_track())
        asyncio.create_task(self._read_video_track())
    
    def bind(self, rtc: RTCPeerConnection):
        rtc.addTrack(self.video_out)
        rtc.addTrack(self.audio_out)
        
        @rtc.on("track")
        async def on_track(track: MediaStreamTrack):
            if track.kind == "video":
                self.video_in = cast(VideoStreamTrack, track)
            if track.kind == "audio":
                self.audio_in = cast(AudioStreamTrack, track)
    
    async def _read_audio_track(self):
        resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        while True:
            if self.audio_in is not None:
                frame = resampler.resample(await self.audio_in.recv())[0] # type: ignore
                pcm = frame.to_ndarray() 
                self.stt.accept(pcm.flatten())
                
            await asyncio.sleep(0)

    async def _read_video_track(self):
        while True:
            if self.video_in is not None:
                frame = await self.video_in.recv()
                continue
            await asyncio.sleep(0.1)
                
                
        