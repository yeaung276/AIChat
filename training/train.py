import os
import wandb
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging

from google.colab import userdata

logging.set_verbosity_info()

# ===== Model Configurations =====
BASE_MODEL = "unsloth/tinyllama-bnb-4bit"               # Model name for 4-bit precision loading
# BASE_MODEL = "unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit"    # Model name for 4-bit precision loading
MAX_SEQ_LENGTH = 2048                                   # Maximum sequence length supported by the model
DTYPE = None                                            # Set to None for auto-detection, Float16 for T4/V100, Bfloat16 for Ampere GPUs
LOAD_IN_4BIT = True                                     # Enable 4-bit loading for memory efficiency

# ===== PEFT Configurations =====
LORA_RANK = 16                                          # LoRA rank for tiny-llama, affects the number of trainable parameters
# LORA_RANK = 8                                             # LoRA rank for Qwen, affects the number of trainable parameters
LORA_ALPHA = 2 * LORA_RANK
LORA_DROPOUT = 0.05                                        # Dropout for regularization, currently set to 0
# TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj"]  # LoRA modules for Qwen
TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]                                                   # LoRA module for tiny-llama

# ===== Dataset Configurations =====
TRAIN_DATASET = "train.jsonl"
EVAL_DATASET = "dev.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/"

# ===== Training Configurations =====
PER_DEVICE_TRAIN_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
MAX_SEQ_LENGTH = 2048
# LEARNING_RATE = 3e-4                # LR for qwen
LEARNING_RATE = 5e-5                # LR for tiny-llama
NUM_TRAIN_EPOCHS = 1
MAX_STEPS = -1                      # Train for full epochs
LOGGING_STEPS = 50
EVAL_STRATEGY = "steps"
EVAL_STEPS = 100
SAVE_STEPS = 400

# ===== Telemetry Configurations =====
WB_PROJECT_NAME = "tiny-llama"
WB_RUN_NAME = "unsloth-tiny-llama-2"


# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# Convert the model for PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load the dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_DATASET,
        "validation": EVAL_DATASET,
    },
)
dataset = dataset.select_columns(["text"])

# Telemetry configuration
wandb.login(key=userdata.get('WB_SECRET'), anonymous="must", force=True)

# Training setup
train_args = TrainingArguments(
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_strategy=EVAL_STRATEGY,
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    warmup_steps=300,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    max_steps=MAX_STEPS,
    logging_steps=LOGGING_STEPS,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    report_to="wandb",
    output_dir=os.path.join(OUTPUT_DIR, WB_RUN_NAME),
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    args=train_args,
)

# Run training
with wandb.init(project=WB_PROJECT_NAME, name=WB_RUN_NAME):
    trainer.train()

# Final model already saved in drive via output_dir, but this ensures tokenizer is there too
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, WB_RUN_NAME))