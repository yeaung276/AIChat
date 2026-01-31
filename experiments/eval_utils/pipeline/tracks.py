import cv2
import librosa
import asyncio
import fractions
import numpy as np
import soundfile as sf

from aiortc import MediaStreamTrack
from av.audio.frame import AudioFrame
from av.video.frame import VideoFrame

class MockAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate=48000, channels=1, frame_samples=960):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_samples = frame_samples

        self._pcm = None
        self._cursor = 0
        self._pts = 0
        self._time_base = fractions.Fraction(1, sample_rate)

    def load_audio(self, path: str):
        pcm, sr = sf.read(path, dtype="float32")

        if pcm.ndim == 1:
            pcm = pcm[:, None]

        if sr != self.sample_rate:
            pcm = librosa.resample(
                pcm.T,
                orig_sr=sr,
                target_sr=self.sample_rate,
            ).T

        self._pcm = pcm
        self._cursor = 0

    async def recv(self) -> AudioFrame:
        await asyncio.sleep(self.frame_samples / self.sample_rate)

        if self._pcm is None or self._cursor >= len(self._pcm):
            samples = np.zeros((self.frame_samples, self.channels), dtype="float32")
        else:
            end = self._cursor + self.frame_samples
            samples = self._pcm[self._cursor:end]
            self._cursor = end

            if len(samples) < self.frame_samples:
                pad = np.zeros(
                    (self.frame_samples - len(samples), self.channels),
                    dtype="float32",
                )
                samples = np.vstack([samples, pad])

        frame = AudioFrame.from_ndarray(
            samples.T,
            format="flt",
            layout="mono" if self.channels == 1 else "stereo",
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base

        self._pts += self.frame_samples
        return frame
    
    
class MockVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, fps: int = 30, width: int = 640, height: int = 480):
        super().__init__()
        self.fps = fps
        self.width = width
        self.height = height

        self._frame = None
        self._pts = 0
        self._time_base = fractions.Fraction(1, fps)

        # pre-allocate black frame (silence equivalent)
        black = np.zeros((height, width, 3), dtype=np.uint8)
        self._black_frame = VideoFrame.from_ndarray(black, format="bgr24")

    def load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")

        img = cv2.resize(img, (self.width, self.height))
        self._frame = VideoFrame.from_ndarray(img, format="bgr24")


    async def recv(self) -> VideoFrame:
        await asyncio.sleep(1 / self.fps)

        src = self._frame if self._frame is not None else self._black_frame

        frame = src.reformat(
            width=self.width,
            height=self.height,
            format="yuv420p",
        )
        frame.pts = self._pts
        frame.time_base = self._time_base

        self._pts += 1
        return frame