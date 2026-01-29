import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from experiments.eval_utils.llm import (
    evaluate_perplexity,
    evaluate_bertscore,
    evaluate_emotion,
    generate,
    get_dataloader,
    evaluate_latency,
)

TEST_DATA = "data/text/test.jsonl"
OUTPUT_DIR = "reports/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    {
        "name": "Base TinyLLaMA",
        "hf_id": "unsloth/tinyllama-chat-bnb-4bit",
    },
    {"name": "Fine-tuned", "hf_id": "unsloth/tinyllama-chat-bnb-4bit", "lora": ""},
]


def load_model(model_id: str, lora=""):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float32, load_in_4bit=True
    ).to(
        DEVICE  # type: ignore
    )

    if lora:
        model = PeftModel.from_pretrained(model, lora)

    model.eval()
    return model, tokenizer


def plot_results(df: pd.DataFrame):
    # ---- Perplexity ----
    plt.figure()
    plt.bar(df["model"], df["perplexity"])
    plt.ylabel("Perplexity (↓ better)")
    plt.title("Fluency Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/perplexity.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- BERTScore F1 ----
    plt.figure()
    plt.bar(df["model"], df["bertscore_f1"])
    plt.ylabel("BERTScore F1 (↑ better)")
    plt.title("Semantic Adequacy Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bertscore_f1.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Emotion Macro-F1 ----
    plt.figure()
    plt.bar(df["model"], df["emotion_macro_f1"])
    plt.ylabel("Emotion Macro-F1 (↑ better)")
    plt.title("Emotional Alignment Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/emotion_macro_f1.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Latency Profile (Grouped Box Plot) ----
    model_names = df["model"].tolist()
    num_models = len(model_names)

    # Extract latency data
    ttft_data = [row["ttft_latency"] for _, row in df.iterrows()]
    e2e_data = [row["e2e_latency"] for _, row in df.iterrows()]

    # Prepare grouped box plot data
    all_data = []
    all_positions = []
    colors = []

    group_width = 3
    box_width = 0.8

    for i in range(num_models):
        base_pos = i * group_width

        # TTFT box (left)
        all_data.append(ttft_data[i])
        all_positions.append(base_pos - 0.5)
        colors.append("#3498db")  # Blue for TTFT

        # E2E box (right)
        all_data.append(e2e_data[i])
        all_positions.append(base_pos + 0.5)
        colors.append("#e74c3c")  # Red for E2E

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plots
    bp = ax.boxplot(
        all_data,
        positions=all_positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        medianprops=dict(color="black", linewidth=2),
        meanprops=dict(
            marker="D", markerfacecolor="gold", markeredgecolor="black", markersize=5
        ),
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set x-axis
    group_centers = [i * group_width for i in range(num_models)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(model_names, fontsize=11)

    # Labels and title
    ax.set_ylabel("Latency (seconds)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Models", fontsize=12, fontweight="bold")
    ax.set_title(
        "Inference Latency Comparison: TTFT vs E2E",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Grid
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.7, label="Time to First Token (TTFT)"),
        Patch(facecolor="#e74c3c", alpha=0.7, label="End-to-End Latency (E2E)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/latency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Trade-off: Perplexity vs Emotion ----
    plt.figure()
    plt.scatter(df["perplexity"], df["emotion_macro_f1"], s=100)
    for i in range(len(df)):
        plt.text(
            df["perplexity"][i],
            df["emotion_macro_f1"][i],
            df["model"][i],
            fontsize=9,
            ha="right",
            va="bottom",
        )

    plt.xlabel("Perplexity (↓ better)")
    plt.ylabel("Emotion Macro-F1 (↑ better)")
    plt.title("Fluency vs Emotional Alignment Trade-off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Trade-off: Latency vs BERTScore ----
    import numpy as np

    # Calculate mean latencies
    mean_e2e_latencies = [np.mean(row["e2e_latency"]) for _, row in df.iterrows()]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        mean_e2e_latencies,
        df["bertscore_f1"],
        s=150,
        alpha=0.7,
        c="#9b59b6",
        edgecolors="black",
        linewidth=1.5,
    )

    for i in range(len(df)):
        plt.text(
            mean_e2e_latencies[i],
            df["bertscore_f1"][i],
            df["model"][i],
            fontsize=10,
            ha="left",
            va="bottom",
            fontweight="bold",
        )

    plt.xlabel("Mean E2E Latency (seconds) (↓ better)", fontsize=12, fontweight="bold")
    plt.ylabel("BERTScore F1 (↑ better)", fontsize=12, fontweight="bold")
    plt.title(
        "Latency vs Semantic Quality Trade-off", fontsize=14, fontweight="bold", pad=15
    )
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/latency_bertscore_tradeoff.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        "Plots saved: perplexity.png, bertscore_f1.png, emotion_macro_f1.png, latency_comparison.png, tradeoff.png, latency_bertscore_tradeoff.png"
    )


def evaluate():
    results = []

    for cfg in MODELS:
        print(f"\n=== Evaluating {cfg['name']} ===")

        model, tokenizer = load_model(cfg["hf_id"], cfg.get("lora", ""))

        dataloader = get_dataloader(TEST_DATA, max_samples=2000, batch_size=200)
        latency_dataloader = get_dataloader(TEST_DATA, max_samples=1000, batch_size=1)

        preds, refs, emotions = generate(model, tokenizer, dataloader)

        # Perplexity (teacher-forced)
        ppl = evaluate_perplexity(
            model,
            tokenizer,
            dataloader,
        )

        # Latency
        latency = evaluate_latency(model, tokenizer, latency_dataloader)

        # BERTScore
        bs = evaluate_bertscore(preds, refs)

        # Emotion Macro-F1
        emotion_f1 = evaluate_emotion(preds, emotions)

        results.append(
            {
                "model": cfg["name"],
                "perplexity": ppl,
                "bertscore_f1": bs["bertscore_f1"],
                "bertscore_p": bs["bertscore_p"],
                "bertscore_r": bs["bertscore_r"],
                "emotion_macro_f1": emotion_f1,
                "ttft_latency": latency["ttft"],
                "e2e_latency": latency["e2e"],
            }
        )

        # Free memory between models
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(df[["model", "perplexity", "bertscore_f1", "emotion_macro_f1"]])

    plot_results(df)


if __name__ == "__main__":
    evaluate()
