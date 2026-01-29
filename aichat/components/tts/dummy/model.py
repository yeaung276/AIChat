from typing import Protocol, AsyncGenerator, Dict, Any


class DummyTTS:
    def __init__(self, *args, **kwargs):
        ...
    
    async def synthesize(self, text: str):
        yield "x0F0F0F", {
            "words": [],
            "wtimes": [],
            "wdurations": [],
            "visames": [],
            "vtimes": [],
            "vdurations": [],
        }

    def configure(*args, **kwargs):
        pass
