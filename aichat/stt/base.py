from typing import Protocol, Literal

class STT(Protocol):
    
    def configure(self, sr: int,  device: Literal['cpu', 'cuda'] = 'cpu'): ...
    
    def accept_chunk(self): ...
    