import time
from typing import Literal


class DummySTT:
    """Generate a dummy string every 20 seconds."""
    
    @classmethod
    def configure(cls, **kwargs):
        pass

    def __init__(self, interval_sec: float = 10.0, **kwargs):
        self.interval = interval_sec
        self._last_emit = 0.0
        self.sample_rate= 16_000

    async def accept(self, samples, sample_rate: int):
        now = time.time()

        # If enough time has passed, emit a string and reset timer
        if now - self._last_emit >= self.interval:
            self._last_emit = now
            return "Hello, my name is Andrew and I live in Bangkok, thailand"

        return None