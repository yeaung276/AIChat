import time
from typing import Literal


class DummySTT:
    """Generate a dummy string every 20 seconds."""

    def __init__(self, interval_sec: float = 20.0):
        self.interval = interval_sec
        self._last_emit = 0.0

    async def accept(self, samples):
        now = time.time()

        # If enough time has passed, emit a string and reset timer
        if now - self._last_emit >= self.interval:
            self._last_emit = now
            return "This is output from dummy stt component"

        # Otherwise return nothing
        return None