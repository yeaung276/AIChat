# Run this script: uv run python -m training.data
import re
import logging

from datasets import load_dataset
from transformers import AutoTokenizer

from aichat.utils.prompt import build_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVE_PATH = "data"
CSV_PATH = "data/emotion-emotion_69k 2.csv"
TOKENIZER = "unsloth/tinyllama-bnb-4bit"
TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VAL_SIZE = 0.1


def formatting_prompts_func(sample):
    dialogue = sample["empathetic_dialogues"]
    dialogue = re.sub(r"\bCustomer\s*:\s*", "", dialogue)
    dialogue = re.sub(r"\bAgent\s*:\s*", "", dialogue)
    dialogue = dialogue.strip()

    text = (
        build_prompt(
            situation=sample["Situation"],
            emotion=sample["emotion"],
            user=dialogue,
            agent=sample["labels"],
        )
        + tokenizer.eos_token
    )

    return {"text": text, "empathetic_dialogues": dialogue}



# ======================= Core ================================
logger.info("loading csv file from %s ...", CSV_PATH)
raw = load_dataset("csv", data_files=CSV_PATH)["train"]

logger.info("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)

logger.info("transforming data into prompt format...")
raw = raw.map(formatting_prompts_func)  # type: ignore

logger.info("splitting dataset with %d train, %d val and %d test size...", TRAIN_SIZE, VAL_SIZE, TEST_SIZE)
first_split = 1 - TRAIN_SIZE
splits = raw.train_test_split(test_size=first_split, seed=42)  # type: ignore

temp_splits = splits["test"].train_test_split(test_size=(TEST_SIZE/first_split), seed=42)

train_dataset = splits["train"]
dev_dataset = temp_splits["train"]
test_dataset = temp_splits["test"]

logger.info("saving data as jsonl to %s...", SAVE_PATH)
train_dataset.to_json(
    "data/train.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

dev_dataset.to_json(
    "data/dev.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

test_dataset.to_json(
    "data/test.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)
