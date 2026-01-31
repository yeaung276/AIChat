import time
import asyncio
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from aichat.pipeline.processor import Processor
from aichat.pipeline.factory import ModelFactory

from experiments.eval_utils.pipeline import Feeder, MockContext, extract_latencies

MAX_DATAPOINT_SAMPLED = 5
AUDIO_DATA = "data/audio/"
VIDEO_DATA = "data/image/"
OUTPUT_DIR = "reports/"

ModelFactory.configure(
    config={
        "speech": [
            {
                "name": "dummy",
                "path": "aichat.components.stt.dummy:DummySTT",
                "config": {},
            }
        ],
        "video": [
            {
                "name": "dummy",
                "path": "aichat.components.video.dummy:DummyVideoAnalyzer",
                "config": {},
            }
        ],
        "llm": [
            {
                "name": "dummy",
                "path": "aichat.components.llm.dummy:DummyLLM",
                "config": {},
            }
        ],
        "tts": [
            {
                "name": "dummy",
                "path": "aichat.components.tts.dummy:DummyTTS",
                "config": {},
            }
        ],
        "avatars": {
            "voices": [{"name": "test_voice", "path": "/test/voice.mp3"}],
            "faces": [
                {"name": "test_face", "path": "/test/face.png", "gender": "neutral"}
            ],
        },
    }
)

feeder = Feeder()

def report(df: pd.DataFrame):
    # ---------- TABLE ----------
    summary = (
        df.groupby("component")["latency_ms"]
        .agg(["mean", "median", "std", "max"])
        .sort_values("mean", ascending=False)
    )

    print("\n=== LATENCY SUMMARY (ms) ===")
    print(summary)

    # ---------- BOTTLENECK ----------
    bottleneck = summary.index[0]
    print(f"\n🔥 BOTTLENECK: {bottleneck}")

    # ---------- PLOT 1: latency over time ----------
    plt.figure()
    for comp in df["component"].unique():
        subset = df[df["component"] == comp]
        plt.plot(subset["sample"], subset["latency_ms"], label=comp)

    plt.xlabel("Sample")
    plt.ylabel("Latency (ms)")
    plt.title("Latency over Time (Stability)")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/latency_over_time.png", dpi=300)
    plt.close()

    # ---------- PLOT 2: distribution ----------
    plt.figure()
    df.boxplot(column="latency_ms", by="component", rot=30)
    plt.ylabel("Latency (ms)")
    plt.title("Latency Distribution per Component")
    plt.suptitle("")
    plt.savefig(f"{OUTPUT_DIR}/latency_distribution.png", dpi=300)
    plt.close()
    
    
async def evaluate():
    processor = Processor(
        speech="dummy",
        video="dummy",
        llm="dummy",
        tts="dummy",
        voice="af_sky",
        context=MockContext(
            prompt="You are a gail name sky who love to have heart warming conversation."
        ),
    )

    await processor.bind(
        feeder.get_input_device(),
        feeder.get_output_device(),
    )

    await feeder.start(audio_path="data/audio", video_path="data/image")

    records = []
    sample_id = 0

    await feeder.feed_next(use_different_sample=False)

    pbar = tqdm(total=MAX_DATAPOINT_SAMPLED, desc="Profiling pipeline")

    async for out in feeder.output_stream():
        if out.get("type") != "AVATAR_SPEAK":
            continue

        waterfall = out["data"]["waterfall"]
        records.extend(extract_latencies(waterfall, sample_id))

        sample_id += 1
        pbar.update(1)

        if sample_id >= MAX_DATAPOINT_SAMPLED:
            break

        await feeder.feed_next(use_different_sample=False)

    pbar.close()

    df = pd.DataFrame(records)
    report(df)


if __name__ == "__main__":
    asyncio.run(evaluate())
