from typing import Protocol, Literal

from av import VideoFrame

class VideoAnalyzer(Protocol):
    @property
    def emotion(self) -> Literal['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
        ...
    
    async def accept(self, frame: VideoFrame): ...