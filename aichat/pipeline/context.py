from typing import Literal

import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM

from aichat.utils.prompt import build_prompt


tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    dtype=torch.float16,
    device_map="auto",
)
model = torch.compile(model)


class Context:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.messages = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = False
        self._topic_modelling_msg_index = 0

    def _get_window(self, min_words: int = 50) -> list[dict] | None:
        window = []
        for msg in reversed(self.messages[self._topic_modelling_msg_index :]):
            window.insert(0, msg)
            if sum(len(m["message"].split()) for m in window) >= min_words:
                return window
        return None

    def _update_topic(self, window: list[dict]):
        conversation = "\n".join(
            f"{m['actor'].upper()}: {m['message']}" for m in window
        )
        prompt = (
            f"Analyze this conversation and describe in ONE CONCISE SENTENCE the topic being discussed.\n\n"
            f"Examples:\n"
            f"1. User won $200 on a lottery ticket.\n"
            f"2. User got a nice haircut and shaved their beard.\n"
            f"3. User's mom got them a new husky dog.\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"Situation:"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore
        with torch.no_grad():
            outputs = model.generate(  # type: ignore
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        self.prompt = (
            tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            .split("\n")[0]
            .strip()
        )

        self._pending = False

    async def add(self, actor: Literal["user", "assistant"], message: str):
        self.messages.append({"actor": actor, "message": message})

        window = self._get_window()
        if not self._pending and window:
            self._pending = True
            self._topic_modelling_msg_index += len(window)
            asyncio.get_event_loop().run_in_executor(
                self._executor, self._update_topic, window
            )

    async def get_context(self, emotion: str | None, length: str = "medium") -> str:
        return build_prompt(
            situation=self.prompt,
            emotion=emotion,
            user=self.messages[-1]["message"],
            answer_type=length,
        )
