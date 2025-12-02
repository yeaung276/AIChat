from typing import Protocol, Literal

class LLM(Protocol):
    
    def configure(self, device: Literal['cpu', 'cuda'] = 'cpu'): ...
    
    async def generate(self, text: str): ...