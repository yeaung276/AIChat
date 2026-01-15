import re
from datasets import load_dataset
from transformers import AutoTokenizer

CSV_PATH = "./emotion-emotion_69k 2.csv"
TOKENIZER = "unsloth/tinyllama-bnb-4bit"

# Load CSV → DatasetDict({"train": Dataset})
raw = load_dataset("csv", data_files=CSV_PATH)["train"]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)

prompt = """Below is a situation you are in, paired with an input that provides current emotion and further context.
Write an emotional and appropriate response.
### Situation:
{}
### Emotion:
{}
### User:
{}
### You:
{}
"""

def formatting_prompts_func(sample):
    dialogue = sample["empathetic_dialogues"]
    dialogue = re.sub(r"\bCustomer\s*:\s*", "", dialogue)
    dialogue = re.sub(r"\bAgent\s*:\s*", "", dialogue)
    dialogue = dialogue.strip()

    text = prompt.format(
        sample["Situation"],
        sample["emotion"],
        dialogue,
        sample["labels"],
    ) + tokenizer.eos_token

    return {"text": text}

raw = raw.map(formatting_prompts_func) # type: ignore

# ---- SPLITS ----
# 1) train (80%) / temp (20%)
splits = raw.train_test_split(test_size=0.2, seed=42) # type: ignore

# 2) temp → dev (10%) / test (10%)
temp_splits = splits["test"].train_test_split(test_size=0.5, seed=42)

train_dataset = splits["train"]
dev_dataset   = temp_splits["train"]
test_dataset  = temp_splits["test"]

# ---- SAVE ----
train_dataset.to_json(
    "train.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

dev_dataset.to_json(
    "dev.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

test_dataset.to_json(
    "test.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)
