from pydantic import BaseModel
from typing import Optional

class RegisterRequest(BaseModel):
    username: str
    password: str
    
    name: str
    bio: Optional[str]
    

class LoginRequest(BaseModel):
    username: str
    password: str