import pytest

from aichat.components.tts.kokoro import KokoroTTS


class TestKokoroTTS:
    def test_kokoro_implement_methods(self):
        KokoroTTS.configure()

        tts = KokoroTTS()
        tts.generate("Hello?")
        
    def test_kokoro_return_correct_metadata(self):
        KokoroTTS.configure()
        
        tts = KokoroTTS()
        for audio, meta in tts.generate("Hello?"):
            assert len(audio) > 0, "Should return audio bytes."
            assert len(meta["words"]) > 0, "Should have word timings."
            assert len(meta["words"]) == len(meta["wtimes"]), "Should have word timings."
            assert len(meta["words"]) == len(meta["wdurations"]), "Should have word timings."
            assert len(meta["visemes"]) > 0, "Should have visame timings."
            assert len(meta["visemes"]) == len(meta["vtimes"]), "Should have visame timings."
            assert len(meta["visemes"]) == len(meta["vdurations"]), "Should have visame timings."