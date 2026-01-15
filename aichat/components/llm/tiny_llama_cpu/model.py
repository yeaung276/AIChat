import asyncio
from typing import AsyncGenerator

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)


class TinyLLama:
    model = None
    tokenizer = None
    generation_kwargs = None

    @classmethod
    def configure(
        cls,
        model: str = "unsloth/tinyllama-chat-bnb-4bit",
        temperature: float = 0.7,
        max_token: int = 512,
        device: str = "cpu",
    ):
        cls.tokenizer = AutoTokenizer.from_pretrained(model)
        cls.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float32,
        ).to(device) # type: ignore

        cls.model.eval()

        cls.generation_kwargs = {
            "max_new_tokens": max_token,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": cls.tokenizer.eos_token_id,
        }

    async def generate(self, text: str) -> AsyncGenerator[str, None]:
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not configured.")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def _generate():
            self.model.generate( # type: ignore
                **inputs,
                streamer=streamer,
                **self.generation_kwargs, # type: ignore
            )

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _generate)

        for token in streamer:
            yield token

    async def warmup(self, text: str = "Hello"):
        async for _ in self.generate(text):
            break
