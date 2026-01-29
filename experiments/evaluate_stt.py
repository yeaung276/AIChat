import importlib

import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from experiments.eval_utils.speech import (
    normalize_text,
    transcribe,
    evaluate_wer,
    evaluate_rtf,
    evaluate_latency,
    load_dataset,
)

TEST_DATA = "data/audio/"
OUTPUT_DIR = "reports/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    {
        "name": "zipformer",
        "path": "aichat.components.stt.zipformer:ZipformerSTT",
        "args": {"sample_rate": 16000, "model_dir": "models/zipformer"},
    },
    {
        "name": "paraformer",
        "path": "aichat.components.stt.paraformer:ParaformerSTT",
        "args": {"sample_rate": 16000, "model_dir": "models/paraformer"}
    }
]


def load_models():
    models = []
    for m in MODELS:
        module_path, class_name = m["path"].rsplit(":", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model_class.configure(**m["args"])
        models.append({"name": m["name"], "model": model_class()})
    return models

def plot_results(results):
    df = pd.DataFrame(results)
    
    # ---- WER ----
    plt.figure()
    plt.bar(df["model"], df["WER"])
    plt.ylabel("WER (↓ better)")
    plt.title("STT Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stt_wer.png", dpi=300)
    plt.close()

    # ---- RTF ----
    plt.figure()
    plt.bar(df["model"], df["RTF"])
    plt.ylabel("RTF (↓ faster)")
    plt.title("STT Speed Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stt_rtf.png", dpi=300)
    plt.close()

    # ---- Latency Box Plot ----
    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [row["endpoint_latency"]["raw"] for _, row in df.iterrows()]

    ax.boxplot(box_data, labels=df["model"], patch_artist=True) # type: ignore
    ax.set_ylabel("Endpoint Latency (seconds)")
    ax.set_title("STT Endpoint Latency Distribution")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stt_latency.png", dpi=300)
    plt.close()

    # ---- Trade-off: Accuracy vs Speed ----
    plt.figure()
    plt.scatter(df["WER"], df["RTF"])
    for i in range(len(df)):
        plt.text(df["WER"].iloc[i], df["RTF"].iloc[i], df["model"].iloc[i])

    plt.xlabel("WER (↓ better)")
    plt.ylabel("RTF (↓ better)")
    plt.title("Accuracy vs Speed Trade-off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stt_tradeoff.png", dpi=300)
    plt.close()

    print("Saved: stt_wer.png, stt_rtf.png, stt_latency.png, stt_tradeoff.png")

async def evaluate():
    print("Loading dataset...")
    dataset = load_dataset(TEST_DATA)

    print("Loading models...")
    models = load_models()

    results = []
    for cfg in models:
        print(f"\n=== Evaluating {cfg['name']} ===")
        refs = []
        hyps = []
        latencies = []

        total_audio = 0.0
        total_processing = 0.0

        for wav, ref in tqdm(dataset):
            result = await transcribe(cfg["model"], wav)

            refs.append(normalize_text(ref))
            hyps.append(result["text"])

            total_audio += result["audio_duration"]
            total_processing += result["processing_time"]
            latencies.append(result["endpoint_latency"])

        results.append(
            {
                "model": cfg["name"],
                "WER": evaluate_wer(refs, hyps),
                "RTF": evaluate_rtf(total_processing, total_audio),
                "endpoint_latency": evaluate_latency(latencies),
            }
        )
    
    df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(df)

    plot_results(results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(evaluate())
