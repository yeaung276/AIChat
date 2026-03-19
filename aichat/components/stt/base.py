from typing import Protocol, Literal

class STT(Protocol):
    sample_rate: int
    
    @classmethod
    def configure(cls, **kwargs): ...
    
    async def is_speaking(self) -> bool: ...
    
    async def accept(self, samples, sample_rate: int) -> None|str: ...
    