from typing import Protocol, Any
import asyncio
    

class GenerationPipeline:
    def __init__(self, queue_in: asyncio.Queue):
        ...
        
    def accept(self, q: asyncio.Queue):
        ...