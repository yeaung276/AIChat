import re
import io
import wave
from typing import Dict, List, Tuple, Literal

import numpy as np
from kokoro import KPipeline


class VisemeSync:
    """Handles phoneme-to-viseme mapping and timestamp synchronization"""

    # Phoneme to Oculus Viseme mapping
    PHONEME_TO_VISEME = {
        # Vowels
        "a": "aa",
        "ɑ": "aa",
        "ɐ": "aa",
        "ɒ": "aa",
        "æ": "aa",
        "ʌ": "aa",
        "e": "E",
        "ə": "E",
        "ɚ": "RR",
        "ɛ": "E",
        "ɜ": "E",
        "œ": "E",
        "i": "I",
        "ɨ": "I",
        "ɪ": "I",
        "ɩ": "I",
        "ᵻ": "I",
        "o": "O",
        "ɔ": "O",
        "ø": "O",
        "ɤ": "O",
        "u": "U",
        "ɯ": "U",
        "ʊ": "U",
        # Diphthongs
        "A": "E",  # eɪ
        "I": "I",  # aɪ
        "O": "O",  # oʊ
        "Q": "O",  # əʊ
        "W": "U",  # aʊ
        "Y": "I",  # ɔɪ
        # Consonants - Bilabial
        "p": "PP",
        "b": "PP",
        "m": "PP",
        # Consonants - Labiodental
        "f": "FF",
        "v": "FF",
        "β": "FF",
        "ɸ": "FF",
        "ʋ": "FF",
        # Consonants - Dental/Alveolar
        "t": "DD",
        "d": "DD",
        "n": "nn",
        "T": "DD",
        "ɖ": "DD",
        "ɟ": "DD",
        "ɾ": "DD",
        "ʈ": "DD",
        "ɳ": "nn",
        "ɲ": "nn",
        "ɴ": "nn",
        "ŋ": "nn",
        # Consonants - Alveolar/Postalveolar
        "s": "SS",
        "z": "SS",
        "ʃ": "SS",
        "ʒ": "SS",
        "ʂ": "SS",
        "x": "SS",
        "S": "SS",
        "ɕ": "SS",
        "ç": "SS",
        # Consonants - Retroflex/Rhotic
        "r": "RR",
        "l": "RR",
        "ɹ": "RR",
        "ɻ": "RR",
        "ʁ": "RR",
        "ɽ": "RR",
        "ʎ": "RR",
        # Consonants - Velar
        "k": "kk",
        "c": "kk",
        "q": "kk",
        "ɡ": "kk",
        # Consonants - Affricates
        "ʤ": "CH",
        "ʥ": "CH",
        "ʦ": "CH",
        "ʨ": "CH",
        "ʧ": "CH",
        # Consonants - Approximants
        "j": "I",
        "w": "U",
        "ɥ": "U",
        "ɰ": "U",
        "ʝ": "I",
        # Dental fricatives
        "ð": "TH",
        "θ": "TH",
        # Special markers (no viseme)
        "$": None,
        ";": None,
        ":": None,
        ",": None,
        ".": None,
        "!": None,
        "?": None,
        "—": None,
        "…": None,
        '"': None,
        "(": None,
        ")": None,
        '"': None,
        '"': None,
        " ": None,
        "\u0303": None,
        "ᵝ": None,
        "ꭧ": None,
        "ᵊ": None,
        "h": None,
        "ɣ": None,
        "χ": None,
        "ʔ": None,
        "ˈ": None,  # Primary stress
        "ˌ": None,  # Secondary stress
        "ː": None,  # Length marker
        "ʰ": None,  # Aspiration
        "ʲ": None,  # Palatalization
        "↓": None,
        "→": None,
        "↗": None,
        "↘": None,
    }

    # Timing constants (ms)
    DELTA_START = -10  # Start viseme 10ms early
    DELTA_END = 10  # Hold viseme 10ms longer

    def __init__(self, frame_rate: int = 40):
        """
        Args:
            frame_rate: TTS model frame rate (frames per second)
        """
        self.frame_rate = frame_rate
        self.ms_per_frame = 1000.0 / frame_rate

    def phoneme_to_viseme(self, phoneme: str) -> str | None:
        """Convert a single phoneme to its corresponding viseme"""
        return self.PHONEME_TO_VISEME.get(phoneme)

    def calculate_timestamps(
        self,
        phonemes: str,
        durations: List[float],
        words: List[str],
        word_boundaries: List[int],
    ) -> Dict:
        times = [0.0]
        cumulative = 0.0
        for duration in durations:
            cumulative += self.ms_per_frame * duration
            times.append(round(cumulative))

        # Extract visemes and their timings
        visemes = []
        vtimes = []
        vdurations = []

        for i, phoneme in enumerate(phonemes):
            viseme = self.phoneme_to_viseme(phoneme)

            if viseme:
                # New visible phoneme
                # +1 offset because durations[0] is BOS token
                visemes.append(viseme)
                start_time = times[i + 1] + self.DELTA_START
                end_time = times[i + 2] + self.DELTA_END
                vtimes.append(max(0, start_time))
                vdurations.append(max(0, end_time - start_time))
            elif phoneme == "ː":
                # Length marker - extend previous viseme
                if visemes:
                    end_time = times[i + 2] + self.DELTA_END
                    vdurations[-1] = max(0, end_time - vtimes[-1])

        # Word timings
        wtimes = []
        wdurations = []
        if words and word_boundaries:
            for j, boundary_idx in enumerate(word_boundaries):
                # +1 offset for BOS token
                start_time = times[boundary_idx + 1] + self.DELTA_START
                wtimes.append(max(0, start_time))

                # Calculate end time (next boundary or end)
                if j < len(word_boundaries) - 1:
                    end_idx = word_boundaries[j + 1]
                else:
                    end_idx = len(phonemes)

                end_time = times[end_idx + 1] + self.DELTA_END
                wdurations.append(max(0, end_time - start_time))
        else:
            words = []

        return {
            "words": words,
            "wtimes": wtimes,
            "wdurations": wdurations,
            "visemes": visemes,
            "vtimes": vtimes,
            "vdurations": vdurations,
        }


class KokoroTTS:
    pipeline = None
    frame_rate = 40
    sample_rate = 24_000

    @classmethod
    def configure(
        cls, model="hexgrad/Kokoro-82M", device: Literal["cpu", "cuda"] = "cpu", frame_rate=40, sample_rate=24_000
    ):
        cls.pipeline = KPipeline(lang_code="a", device=device, repo_id=model)
        cls.frame_rate = frame_rate
        cls.sample_rate = sample_rate

    def __init__(self, voice: str = "af_sky", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.viseme_sync = VisemeSync()

    def generate(self, text: str):
        if self.pipeline is None:
            raise Exception("Engine not configured.")

        for r in self.pipeline(text, voice=self.voice):
            words, word_boundaries = self._extract_words(text, r.phonemes)

            metadata = self.viseme_sync.calculate_timestamps(
                phonemes=r.phonemes,
                durations=r.pred_dur.tolist(), # type: ignore
                words=words,
                word_boundaries=word_boundaries,
            )

            yield self.encode_wav(r.audio.numpy(), sample_rate=self.sample_rate), metadata # type: ignore

    def _extract_words(
        self, text: str, phonemes: str
    ) -> Tuple[List[str], List[int]]:
        words = re.findall(r"\b\w+\b", text)

        if words and phonemes:
            phonemes_per_word = len(phonemes) / len(words)
            boundaries = [int(i * phonemes_per_word) for i in range(len(words))]
        else:
            boundaries = []

        return words, boundaries

    def encode_wav(self, samples, sample_rate: int) -> bytes:
        samples_int16 = (samples * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples_int16.tobytes())

        return buffer.getvalue()
