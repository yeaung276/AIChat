import logging
import os
import asyncio
from typing import Any

from .tracks import MockAudioStreamTrack, MockVideoStreamTrack

class Feeder:
    def __init__(self):
        self._track_handler = None
        self._out_queue = asyncio.Queue()
        
        self._audio = []
        self._video = []
        self._index = 0
        
        self.audio_track = None
        self.video_track = None
        
    def get_input_device(self) -> Any:
        feeder = self
        class RTC:
            def on(self,  *args, **kwargs):
                def decorator(fn):
                    feeder._track_handler = fn
                    return fn
                return decorator
                
        return RTC()
                
        
    def get_output_device(self) -> Any:
        feeder = self
        class WS:
            async def send_json(self, obj):
                return await feeder._out_queue.put(obj)
        return WS()
        
    async def start(self, audio_path: str, video_path: str):
        logging.info("Reading files in the directory...")
        self._audio = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith(".flac")]
        self._video = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".jpg")]
        
        
        self.audio_track = MockAudioStreamTrack()
        self.video_track = MockVideoStreamTrack()
        
        logging.info("Starting tracking handlers...")
        if self._track_handler:
            await self._track_handler(self.audio_track) # type: ignore
            await self._track_handler(self.video_track) # type: ignore

        
    async def feed_next(self, use_different_sample = True):
        logging.info("Feeding next audio...")
        if self.audio_track:
            self.audio_track.load_audio(self._audio[self._index % len(self._audio)])
        if self.video_track:
            self.video_track.load_image(self._video[self._index % len(self._video)])
        if use_different_sample:
            self._index += 1

    async def output_stream(self):
        while True:
            yield await self._out_queue.get()
            
    
    