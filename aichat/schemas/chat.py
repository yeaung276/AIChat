from typing import Literal

from pydantic import BaseModel

class Agent(BaseModel):
    voice: Literal["af_bella"] = "af_bella"
    face: Literal["julia"] = "julia"
    prompt: str = ""
    

class ChatRequest(BaseModel):
    agent: Agent
    speech: str | None = "zipformer"
    dialogue: str | None = "tiny_llama"
    
    