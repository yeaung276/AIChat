from typing import Protocol, Literal

class STT(Protocol):
    def accept(self, samples) -> None|str: ...
    