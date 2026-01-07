import time
import math
import asyncio
from typing import Literal
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from av import VideoFrame 
from deepface import DeepFace

class DeepFaceVideoAnalyzer:
    _executor = ThreadPoolExecutor(max_workers=1)
    decay_rate = 7
    interval = 0.1
    
    @classmethod
    def configure(cls, decay_rate: int = 7, sample_rate: int = 10):
        dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(
            img_path=dummy_face,
            actions=["emotion"],
            enforce_detection=False
        )
        cls.decay_rate = decay_rate
        cls.interval = 1 / sample_rate
    
        
    @property
    def emotion(self) -> Literal['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
        return max(self._scores, key=self._scores.get) # type: ignore
    
    def __init__(self):
        self._latest_frame = None
        self._lock = asyncio.Lock()
        self._scores = {
            "angry": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "surprise": 0.0,
            "neutral": 0.0,
        }
        self._last_update = time.monotonic()
        self._task = asyncio.create_task(self._inference_loop())
        
    async def accept(self, frame: VideoFrame):
        async with self._lock:
            self._latest_frame = frame
        
    async def _inference_loop(self):
        loop = asyncio.get_running_loop()
        next_pred = time.monotonic()
        
        while True:
            now = time.monotonic()
            if now < next_pred:
                await asyncio.sleep(next_pred - now)
                
            async with self._lock:
                if self._latest_frame is None:
                    await asyncio.sleep(0.01)
                    continue
                img = self._latest_frame.to_ndarray(format="bgr24")
            
        
            try:
                emotion = await loop.run_in_executor(
                    self._executor, 
                    lambda img: DeepFace.analyze(img, actions=["emotion"], enforce_detection=False), # type: ignore
                    img
                )
                await self._update_scores(emotion[0]['emotion']) # type: ignore
                next_pred = time.monotonic() + self.interval
            except Exception as e:
                print("DeepFace error:", e)
            
            await asyncio.sleep(0)
            
    async def _update_scores(self, probs):
        async with self._lock:
            now = time.monotonic()
            dt = now - self._last_update
            self._last_update = now
        
        alpha = 1.0 - math.exp(-self.decay_rate * dt)
        
        for k in self._scores:
            prev = self._scores[k]
            obs = probs.get(k, 0.0)
            self._scores[k] = (1 - alpha) * prev + alpha * obs