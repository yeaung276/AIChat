from typing import Protocol, AsyncGenerator, Dict, Any


class DummyTTS:
    def __init__(self, *args, **kwargs):
        ...
    
    async def synthesize(self, text: str):
        yield b"\x0f\x0f\x0f", {
            "words": [],
            "wtimes": [],
            "wdurations": [],
            "visames": [],
            "vtimes": [],
            "vdurations": [],
        }

    def configure(*args, **kwargs):
        pass
