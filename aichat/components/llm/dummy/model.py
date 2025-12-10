class DummyLLM:        
    async def generate(self, text: str):
        yield "hello this is response from dummy llm"
    
    async def warmup(self, text: str):
        pass
    