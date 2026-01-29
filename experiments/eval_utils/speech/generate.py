import re
import time
import asyncio
from tqdm.asyncio import tqdm

import numpy as np
import soundfile as sf
from av import AudioFrame
from av.audio.resampler import AudioResampler

CHUNK_SIZE = 1600  # 100ms @ 16kHz


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def webrtc_audio_frames(
    wav_path: str,
    frame_ms: int = 20,
):
    audio, sr = sf.read(wav_path, dtype="int16")
    audio = audio.reshape(1, -1)  # mono

    samples_per_frame = int(sr * frame_ms / 1000)

    for i in range(0, len(audio), samples_per_frame):
        chunk = audio[i : i + samples_per_frame]
        frame = AudioFrame.from_ndarray(chunk, format="s16", layout="mono")
        frame.sample_rate = sr

        yield frame


async def transcribe(stt, wav_path: str):
    resampler = AudioResampler(
        format="s16",
        layout="mono",
        rate=stt.sample_rate,
    )

    texts = []
    last_voice_time = time.perf_counter()
    start_wall = time.perf_counter()
    async for frame in tqdm(webrtc_audio_frames(wav_path, stt.sample_rate)):
        resampled = resampler.resample(frame)[0]
        pcm = resampled.to_ndarray().flatten()
        
        if np.any(pcm):
            last_voice_time = time.perf_counter()

        result = await stt.accept(pcm, stt.sample_rate)

        if result:
            texts.append(result)
    
    texts.append(await stt.flush())
    final_time = time.perf_counter()

    return {
        "text": normalize_text(" ".join(texts)),
        "audio_duration": sf.info(wav_path).duration,
        "processing_time": final_time - start_wall,
        "endpoint_latency": final_time - last_voice_time,
    }
