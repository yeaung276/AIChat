import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aichat.components.llm.transformer import Transformer
from aichat.utils.prompt import build_prompt

INPUTS = [
    ("happy",   "I just got promoted at work today!"),
    ("sad",     "My dog passed away this morning."),
    ("fearful", "I have a big presentation tomorrow and I can't stop overthinking."),
    ("angry",   "My roommate keeps eating my food without asking."),
    ("neutral", "I've been thinking about learning guitar."),
]

SITUATION = "A warm and empathetic companion who listens and responds with care."


async def main():
    Transformer.configure(
        model="unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit",
        lora_path="models/qwen2.5-lora",
        lora_name="main",
        lora_rank=8,
    )
    llm = Transformer()

    for emotion, user_msg in INPUTS:
        prompt = build_prompt(situation=SITUATION, emotion=emotion, user=user_msg)
        print(f"\n[{emotion}] {user_msg}")
        print(">>> ", end="", flush=True)
        async for token in llm.generate(prompt):
            print(token, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())
