from typing import AsyncGenerator

class TinyLLama:
    engine = None
    params = None

    @classmethod
    def configure(cls, model="unsloth/tinyllama-chat-bnb-4bit", temperature=0.7, max_token=512, device = 'cpu'):
      from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
      engine_args = AsyncEngineArgs(
          model=model,
          max_model_len=max_token,
          enforce_eager=True, 
          gpu_memory_utilization=0.5,
      )
      cls.engine = AsyncLLMEngine.from_engine_args(engine_args)
      cls.params = SamplingParams(temperature=temperature, max_tokens=max_token)
    
    async def generate(self, text: str) -> AsyncGenerator[str, None]:
      if self.engine is None:
        raise Exception("Engine not configured.")
      async for out in self.engine.generate(prompt=text, sampling_params=self.params, request_id="<placeholder>"):
          if out.finished:
                break
          yield out.outputs[0].text
          
    async def warmup(self, text: str):
      pass