# test_audio_utils.py
import av
import numpy as np
from av.audio.resampler import AudioResampler

def webrtc_audio_stream(path: str):
    """
    Load an MP3 file and return a generator yielding PCM int16 mono 16kHz chunks,
    identical to what Processor._read_audio_track sends to stt.accept().
    """
    resampler = AudioResampler(
        format='s16',
        layout='mono',
        rate=16000
    )

    container = av.open(path)
    stream = container.streams.audio[0]

    for frame in container.decode(stream):
        frames = resampler.resample(frame)
        if not frames:
            continue

        for f in frames:
            pcm = f.to_ndarray()      # shape (samples, 1)
            pcm = pcm.flatten()       # shape (samples,)
            pcm = pcm.astype(np.int16)
            yield pcm