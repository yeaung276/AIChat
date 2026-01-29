from typing import AsyncGenerator


class Transformer:
    engine = None
    params = None

    @classmethod
    def configure(
        cls,
        model="unsloth/tinyllama-chat-bnb-4bit",
        temperature=0.7,
        max_token=512,
        gpu_utilization=0.5,
        lora_path: str | None = None,
        lora_name: str | None = None,
        lora_rank: int | None = None,
    ):
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        from vllm.lora.request import LoRARequest

        engine_args = AsyncEngineArgs(
            model=model,
            max_model_len=max_token,
            enforce_eager=True,
            gpu_memory_utilization=gpu_utilization,
            enable_lora=True if lora_path else False,
            max_lora_rank=lora_rank,
        )

        cls.engine = AsyncLLMEngine.from_engine_args(engine_args)
        cls.params = SamplingParams(temperature=temperature, max_tokens=max_token)

        if lora_path:
            cls.lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=1,
                lora_path=lora_path,
            )

    async def generate(self, text: str) -> AsyncGenerator[str, None]:
        if self.engine is None:
            raise Exception("Engine not configured.")

        async for out in self.engine.generate(
            prompt=text,
            sampling_params=self.params,
            request_id="<placeholder>",
            lora_request=self.lora_request,
        ):
            if out.finished:
                break
            yield out.outputs[0].text
    
    async def warmup(self, text: str):
        if self.engine is None:
            raise Exception("Engine not configured.")
        
        async for _ in self.generate(text):
            break