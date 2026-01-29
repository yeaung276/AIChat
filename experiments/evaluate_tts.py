import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from experiments.eval_utils.llm.dataloader import load_single_text_column
from experiments.eval_utils.synthesis.generate import generate


from aichat.components.tts.kokoro import KokoroTTS

TEST_DATA = "data/text/test.jsonl"
OUTPUT_DIR = "reports/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def plot(df):
    # ===== FIGURE 1: Latency vs Input Length =====
    plt.figure()
    plt.scatter(df["text_len_words"], df["latency"], alpha=0.7)
    z = np.polyfit(df["text_len_words"], df["latency"], 1)
    plt.plot(df["text_len_words"], np.poly1d(z)(df["text_len_words"]))
    plt.xlabel("Input Length (words)")
    plt.ylabel("Latency (seconds)")
    plt.title("TTS Latency vs Input Length")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "latency_vs_input_length.png")
    plt.close()
    
    # ===== FIGURE 2: RTF Distribution =====
    plt.figure()
    plt.boxplot(df["rtf"].dropna(), vert=False)
    plt.xlabel("Real-Time Factor (RTF)")
    plt.title("RTF Distribution (Lower is Better)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "rtf_distribution.png")
    plt.close()
    
    # ===== FIGURE 3: Latency Summary =====
    stats = {
        "Mean": df["latency"].mean(),
        "P50": df["latency"].quantile(0.5),
        "P95": df["latency"].quantile(0.95),
    }

    plt.figure()
    plt.bar(stats.keys(), stats.values()) # type: ignore
    plt.ylabel("Latency (seconds)")
    plt.title("TTS Inference Latency Summary")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "latency_summary.png")
    plt.close()


def evaluate():
    results = []

    texts = load_single_text_column(TEST_DATA, column="labels", max_samples=10)

    KokoroTTS.configure()
    tts = KokoroTTS()

    for text in tqdm(texts):
        a_dur, p_dur = generate(tts, text)  # type: ignore
        results.append(
            {
                "text_len_words": len(text.split(" ")),
                "latency": p_dur,
                "rtf": p_dur / a_dur,
            }
        )
        
    df = pd.DataFrame(results)
    plot(df)



if __name__ == "__main__":
    evaluate()
