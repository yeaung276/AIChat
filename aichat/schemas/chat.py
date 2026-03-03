from typing import Literal, Optional

from pydantic import BaseModel

class Agent(BaseModel):
    voice: Literal["af_bella"] = "af_bella"
    face: Literal["julia"] = "julia"
    prompt: str = ""
    

class ChatRequest(BaseModel):
    agent: Agent
    name: str


class FeedbackRequest(BaseModel):
    session_id: int
    q1: int
    q2: int
    q3: int
    q4: int
    q5: Optional[str] = ""
