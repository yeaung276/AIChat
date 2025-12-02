import asyncio
import librosa
import numpy as np
from aiortc import AudioStreamTrack, MediaStreamError
from av.audio.frame import AudioFrame
from fractions import Fraction


class MP3AudioStreamTrack(AudioStreamTrack):
    """
    Emulates a WebRTC audio stream by reading from an MP3 file using librosa.
    Mimics WebRTC properties: 48kHz sample rate, 20ms frames.
    """
    
    def __init__(self, file_path: str, sample_rate: int = 48000, channels: int = 2):
        super().__init__()
        self.target_sample_rate = sample_rate
        self.target_channels = channels
        self.samples_per_frame = int(sample_rate * 0.02)  # 20ms frames
        
        self.audio, _ = librosa.load(
            file_path,
            sr=sample_rate,
            mono=(channels == 1)
        )
        
        # Convert to stereo if needed
        if channels == 2 and self.audio.ndim == 1:
            self.audio = np.stack([self.audio, self.audio])
        elif channels == 2:
            self.audio = self.audio[:2]  # Take first 2 channels
        
        # Convert to int16 to denormalize
        self.audio = (self.audio * 32767).astype(np.int16)
        
        self._position = 0
        self._timestamp = 0
        self._start = None # type: ignore
        
    async def recv(self) -> AudioFrame:
        """Receive the next audio frame, matching WebRTC timing."""
        
        if self._position + self.samples_per_frame > self.audio.shape[1]:
            raise MediaStreamError  # signals end of media
        
        # Extract frame samples
        frame_data = self.audio[:, self._position:self._position + self.samples_per_frame]
    
        self._position += self.samples_per_frame
        
        # Create AudioFrame
        new_frame = AudioFrame.from_ndarray(
            frame_data,
            format='s16',
            layout='mono' if self.target_channels == 1 else 'stereo'
        )
        new_frame.sample_rate = self.target_sample_rate
        new_frame.pts = self._timestamp
        new_frame.time_base = Fraction(1, self.target_sample_rate)
        
        self._timestamp += self.samples_per_frame
        
        # Simulate real-time playback
        if self._start is None:
            self._start = asyncio.get_event_loop().time()
        
        wait_time = (self._timestamp / self.target_sample_rate) - (
            asyncio.get_event_loop().time() - self._start
        )
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        return new_frame
