from typing import Literal
from fastapi import WebSocket

import spacy
import torch
import asyncio
from sqlmodel import Session, update
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM

from aichat.db_models.chat import Chat
from aichat.db_models.db import engine
from aichat.types import MESSAGE_TYPE_TRANSCRIPT
from aichat.utils.prompt import build_prompt

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)
model = torch.compile(model)

class Context:
    def __init__(self, chat: Chat, ws: WebSocket):
        self.chat_id = chat.id
        self.prompt = chat.prompt
        self.messages = chat.transcripts
        self.safe_messages = chat.transcripts.copy()
        self.ws = ws
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = False

    def _get_window(self, min_words: int = 50) -> list[dict] | None:
        window = []
        for msg in reversed(self.messages):
            window.insert(0, msg)
            if sum(len(m["message"].split()) for m in window) >= min_words:
                return window
        return None

    def _housekeeping(self, window: list[dict]):
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

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("generating...")
        with torch.no_grad():
            outputs = model.generate(
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

        for msg in self.messages[len(self.safe_messages) :]:
            doc = nlp(msg["message"])
            result = msg["message"]
            for ent in reversed(doc.ents):
                if ent.label_ in {"PERSON", "GPE", "LOC", "ORG"}:
                    result = (
                        result[: ent.start_char]
                        + f"[{ent.label_}]"
                        + result[ent.end_char :]
                    )
            self.safe_messages.append({"actor": msg["actor"], "message": result})

        with Session(engine) as session:
            session.exec(
                update(Chat)
                .where(Chat.id == self.chat_id) # type: ignore
                .values(transcripts=self.safe_messages, prompt=self.prompt)
            )
            session.commit()

        self._pending = False

    async def add(self, actor: Literal["user", "assistant"], message: str):
        self.messages.append({"actor": actor, "message": message})
        await self.ws.send_json(
            {
                "type": MESSAGE_TYPE_TRANSCRIPT,
                "data": {"actor": actor, "message": message},
            }
        )

        if not self._pending:
            window = self._get_window()
            if window:
                self._pending = True
                asyncio.get_event_loop().run_in_executor(
                    self._executor, self._housekeeping, window
                )

    async def get_context(self, emotion: str) -> str:
        return build_prompt(
            situation=self.prompt, emotion=emotion, user=self.messages[-1]["message"]
        )
