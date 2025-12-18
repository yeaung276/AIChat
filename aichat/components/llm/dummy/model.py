class DummyLLM:        
    async def generate(self, text: str):
        yield "Hi, I am Aura. What can I help you with?"
    
    async def warmup(self, text: str):
        pass
    