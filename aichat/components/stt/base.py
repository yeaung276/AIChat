from typing import Protocol, Literal

class STT(Protocol):
    sample_rate: int
    async def accept(self, samples, sample_rate: int) -> None|str: ...
    