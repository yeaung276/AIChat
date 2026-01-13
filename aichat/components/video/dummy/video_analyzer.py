from typing import Literal

class DummyVideoAnalyzer:
    @property
    def emotion(self) -> Literal["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
        return "happy"
    
    @classmethod
    async def configure(cls, **kwargs):
        pass

    async def accept(self, frame):
        return
        