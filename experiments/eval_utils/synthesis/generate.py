import io
import time
import soundfile as sf
import asyncio

def generate(model, text):
    async def _gen():
        start = time.perf_counter()
        async for audio, _ in model.synthesize(text):
            return audio, time.perf_counter() - start

    audio, dur = asyncio.run(_gen()) # type: ignore
    return sf.info(io.BytesIO(audio)).duration, dur
    