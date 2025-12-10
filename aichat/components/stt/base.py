from typing import Protocol, Literal

class STT(Protocol):
    async def accept(self, samples) -> None|str: ...
    