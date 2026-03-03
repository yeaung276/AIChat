from sqlmodel import SQLModel, Field, Column, JSON


class Chat(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    
    name: str
    voice: str
    face: str
    prompt: str
    
    transcripts: list[dict] = Field(
        sa_column=Column(JSON),
        default_factory=list,
    )

class Feedback(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    
    mean_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    session_duration_s: float
    session_turns: int
    
    q1_rating: int
    q2_rating: int
    q3_rating: int
    q4_rating: int
    q5_answer: str
    
    