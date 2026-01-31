from typing import Literal

from aichat.pipeline.context import Context


class MockContext(Context):
    def __init__(self, prompt):
        self.prompt = prompt
        self.messages = []

    async def add(self, actor: Literal["user", "assistant"], message: str):
        self.messages.append({"actor": actor, "message": message})
