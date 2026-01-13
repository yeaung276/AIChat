from typing import Literal

from pydantic import BaseModel

class Agent(BaseModel):
    voice: Literal["af_bella"] = "af_bella"
    face: Literal["julia"] = "julia"
    prompt: str = ""
    

class ChatRequest(BaseModel):
    agent: Agent
    name: str
    dialogue: str = "tiny_llama"
    
    