import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from experiments.eval_utils.llm import (
    evaluate_perplexity,
    evaluate_bertscore,
    evaluate_emotion,
    generate,
)

TEST_DATA = "experiments/test.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    {
        "name": "Base TinyLLaMA",
        "hf_id": "unsloth/tinyllama-chat-bnb-4bit",
    },
    {
        "name": "Fine-tuned",
        "hf_id": "path/to/your/finetuned-model",
    },
]


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(DEVICE) # type: ignore

    model.eval()
    return model, tokenizer

def plot_results(df: pd.DataFrame):
    # ---- Perplexity ----
    plt.figure()
    plt.bar(df["model"], df["perplexity"])
    plt.ylabel("Perplexity (↓ better)")
    plt.title("Fluency Comparison")
    plt.tight_layout()
    plt.show()

    # ---- BERTScore F1 ----
    plt.figure()
    plt.bar(df["model"], df["bertscore_f1"])
    plt.ylabel("BERTScore F1 (↑ better)")
    plt.title("Semantic Adequacy Comparison")
    plt.tight_layout()
    plt.show()

    # ---- Emotion Macro-F1 ----
    plt.figure()
    plt.bar(df["model"], df["emotion_macro_f1"])
    plt.ylabel("Emotion Macro-F1 (↑ better)")
    plt.title("Emotional Alignment Comparison")
    plt.tight_layout()
    plt.show()

    # ---- Trade-off: Perplexity vs Emotion ----
    plt.figure()
    plt.scatter(df["perplexity"], df["emotion_macro_f1"])
    for i in range(len(df)):
        plt.text(
            df["perplexity"][i],
            df["emotion_macro_f1"][i],
            df["model"][i],
        )

    plt.xlabel("Perplexity (↓ better)")
    plt.ylabel("Emotion Macro-F1 (↑ better)")
    plt.title("Fluency vs Emotional Alignment Trade-off")
    plt.tight_layout()
    plt.show()
    
def evaluate():
    rows = []

    for cfg in MODELS:
        print(f"\n=== Evaluating {cfg['name']} ===")

        model, tokenizer = load_model(cfg["hf_id"])
        
        preds, refs, emotions = generate(
            model,
            tokenizer,
            TEST_DATA,
        )

        # Perplexity (teacher-forced)
        ppl = evaluate_perplexity(
            model,
            tokenizer,
            TEST_DATA,
        )

        # BERTScore 
        bs = evaluate_bertscore(preds, refs)

        # Emotion Macro-F1
        emotion_f1 = evaluate_emotion(preds, emotions)

        rows.append(
            {
                "model": cfg["name"],
                "perplexity": ppl,
                "bertscore_f1": bs["bertscore_f1"],
                "bertscore_p": bs["bertscore_p"],
                "bertscore_r": bs["bertscore_r"],
                "emotion_macro_f1": emotion_f1,
            }
        )

        # Free memory between models
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print("\n=== Final Results ===")
    print(df)

    plot_results(df)



