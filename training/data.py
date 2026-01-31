# Run this script: uv run python -m training.data
import re
import logging

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

from aichat.utils.prompt import build_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVE_PATH = "data/text"
CSV_PATH = "data/text/emotion-emotion_69k 2.csv"
TOKENIZER = "unsloth/tinyllama-bnb-4bit"
TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VAL_SIZE = 0.1

fine_to_basic_emotion = {
    # ANGER 🤬
    "angry": "anger",
    "furious": "anger",
    "annoyed": "anger",
    # DISGUST 🤢
    "disgusted": "disgust",
    "ashamed": "disgust",
    "embarrassed": "disgust",
    # FEAR 😨
    "afraid": "fear",
    "terrified": "fear",
    "anxious": "fear",
    "apprehensive": "fear",
    # JOY 😀
    "joyful": "joy",
    "excited": "joy",
    "proud": "joy",
    "grateful": "joy",
    "hopeful": "joy",
    "content": "joy",
    "impressed": "joy",
    "confident": "joy",
    "anticipating": "joy",
    "prepared": "joy",
    # SADNESS 😭
    "sad": "sadness",
    "lonely": "sadness",
    "guilty": "sadness",
    "disappointed": "sadness",
    "devastated": "sadness",
    "nostalgic": "sadness",
    # SURPRISE 😲
    "surprised": "surprise",
    # NEUTRAL 😐
    "sentimental": "neutral",
    "faithful": "neutral",
    "caring": "neutral",
    "trusting": "neutral",
    "jealous": "neutral",
}


# ======================= Core ================================
logger.info("loading csv file from %s ...", CSV_PATH)
raw = load_dataset("csv", data_files=CSV_PATH)["train"]

logger.info("computing label word lengths...")
raw = raw.map(
    lambda x: {"label_word_len": len(x["labels"].split())}
)

logger.info("computing quantile bins for answer type...")
lengths = np.array(raw["label_word_len"])
q1, q2 = np.quantile(lengths, [0.33, 0.66])

def assign_answer_type(n: int) -> str:
    if n <= q1:
        return "short"
    if n <= q2:
        return "medium"
    return "long"

raw = raw.map(
    lambda x: {"answer_type": assign_answer_type(x["label_word_len"])}
)

logger.info("plotting label length frequency distribution...")

plt.figure(figsize=(10, 6))

# Histogram 
counts, bins, _ = plt.hist(
    lengths,
    bins=50,
    edgecolor="black",
    alpha=0.7,
)

# Quantile markers
plt.axvline(q1, linestyle="--", linewidth=2, color="red", label=f"q33 = {int(q1)}")
plt.axvline(q2, linestyle="--", linewidth=2, color="blue", label=f"q66 = {int(q2)}")

# Bin labels (centered)
y_max = max(counts)

plt.text(q1 / 2, y_max * 0.9, "short", ha="center", fontsize=12, weight="bold")
plt.text((q1 + q2) / 2, y_max * 0.9, "medium", ha="center", fontsize=12, weight="bold")
plt.text((q2 + lengths.max()) / 2, y_max * 0.9, "long", ha="center", fontsize=12, weight="bold")

# Labels / legend
plt.xlabel("Label word length")
plt.ylabel("Frequency")
plt.title("Answer Length Frequency Distribution")
plt.legend()

plt.tight_layout()
plt.savefig(f"{SAVE_PATH}/label_length_distribution.png")
plt.close()

logger.info("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)


def formatting_prompts_func(sample):
    dialogue = sample["empathetic_dialogues"]
    dialogue = re.sub(r"\bCustomer\s*:\s*", "", dialogue)
    dialogue = re.sub(r"\bAgent\s*:\s*", "", dialogue)
    dialogue = dialogue.strip()

    text = (
        build_prompt(
            situation=sample["Situation"],
            emotion=fine_to_basic_emotion.get(sample["emotion"], ""),
            answer_type=sample["answer_type"],
            user=dialogue,
            agent=sample["labels"],
        )
        + tokenizer.eos_token
    )

    return {
        "text": text,
        "empathetic_dialogues": dialogue,
        "emotion": fine_to_basic_emotion.get(sample["emotion"], ""),
        "answer_type": sample["answer_type"],
    }


logger.info("transforming data into prompt format...")
raw = raw.map(formatting_prompts_func)  # type: ignore

logger.info(
    "splitting dataset with %d train, %d val and %d test size...",
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
)
first_split = 1 - TRAIN_SIZE
splits = raw.train_test_split(test_size=first_split, seed=42)  # type: ignore

temp_splits = splits["test"].train_test_split(
    test_size=(TEST_SIZE / first_split), seed=42
)

train_dataset = splits["train"]
dev_dataset = temp_splits["train"]
test_dataset = temp_splits["test"]

logger.info("saving data as jsonl to %s...", SAVE_PATH)
train_dataset.to_json("data/train.jsonl", orient="records", lines=True, force_ascii=False)
dev_dataset.to_json("data/dev.jsonl", orient="records", lines=True, force_ascii=False)
test_dataset.to_json("data/test.jsonl", orient="records", lines=True, force_ascii=False)
