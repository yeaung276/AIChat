import uuid
from typing import AsyncGenerator


class Transformer:
    engine = None
    params = None
    lora_request = None

    @classmethod
    def configure(
        cls,
        model="unsloth/Qwen2.5-0.5B-bnb-4bit",
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
            enable_lora=lora_path is not None,
            max_lora_rank=lora_rank if lora_path else None,
            trust_remote_code=True,
        )

        cls.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        cls.params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_token
        )

        if lora_path:
            cls.lora_request = LoRARequest(
                lora_name=lora_name or "default_lora",
                lora_int_id=1,
                lora_path=lora_path,
            )
        else:
            cls.lora_request = None

    async def generate(self, text: str) -> AsyncGenerator[str, None]:
        if self.engine is None:
            raise Exception("Engine not configured.")

        async for out in self.engine.generate(
            prompt=text,
            sampling_params=self.params,
            request_id=str(uuid.uuid4()),
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