from unsloth import FastLanguageModel
import torch

# Configuration settings
BASE_MODEL = "unsloth/tinyllama-bnb-4bit"       # Model name for 4-bit precision loading
MAX_SEQ_LENGTH = 2048                           # Maximum sequence length supported by the model
DTYPE = None                                    # Set to None for auto-detection, Float16 for T4/V100, Bfloat16 for Ampere GPUs
LOAD_IN_4BIT = True                             # Enable 4-bit loading for memory efficiency
LORA_RANK = 16                                  # LoRA rank, affects the number of trainable parameters
LORA_ALPHA = 16
LORA_DROPOUT = 0                                # Dropout for regularization, currently set to 0

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
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
