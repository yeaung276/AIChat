from sqlmodel import SQLModel, Field, Column, JSON


class Character(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    
    name: str
    voice: str
    face: str
    prompt: str

class ChatSession(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    
    transcripts: list[dict] = Field(
        sa_column=Column(JSON),
        default_factory=list,
    )
    
    mean_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    session_duration_s: float
    session_turns: int
    
    q1_rating: int = 0
    q2_rating: int = 0
    q3_rating: int = 0
    q4_rating: int = 0
    q5_answer: str = ""
    
    