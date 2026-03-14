import sys
import json
import asyncio
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aichat.components.llm.transformer import Transformer
from aichat.pipeline.context import Context

SITUATION = "A warm and empathetic companion who listens and responds with care."
DEFAULT_EMOTION = "neutral"

def get_input(file=None, missing_emotion: str | None=DEFAULT_EMOTION): # return (emotion, text) generator
    if file is not None:
        # [[emotion, text], ...] from json file
        with open(file) as f:
            for emotion, text in json.load(f):
                yield emotion or missing_emotion, text
    else:
        while True:
            try:
                parts = input("You: ").split(":", 1)
                if len(parts) == 2:
                    yield parts[0].strip() or missing_emotion, parts[1].strip()
                else:
                    yield missing_emotion, parts[0].strip()
            except (EOFError, KeyboardInterrupt):
                return

            


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default=None, help="JSON input file [[emotion, text], ...]")
    parser.add_argument("--output", "-o", default="output.json")
    parser.add_argument("--no-emotion", action="store_true", help="Use None when emotion is missing instead of default")
    args = parser.parse_args()

    missing_emotion = None if args.no_emotion else DEFAULT_EMOTION

    Transformer.configure(
        model="unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit",
        lora_path="models/qwen2.5-lora",
        lora_name="main",
        lora_rank=8,
    )

    llm = Transformer()
    ctx = Context(prompt=SITUATION)

    try:
        for emotion, text in get_input(file=args.input, missing_emotion=missing_emotion):
            await ctx.add("user", text)

            response = ""
            async for resp in llm.generate(await ctx.get_context(emotion)):
                response = resp

            await ctx.add("assistant", response)
            print(f"Assistant: {response}")
    finally:
        with open(args.output, "w") as f:
            json.dump(ctx.messages, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
