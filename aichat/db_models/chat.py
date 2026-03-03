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
