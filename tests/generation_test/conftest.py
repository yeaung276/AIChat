import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock
import pytest


def mock_dialogue_core():
    """Mock LLM component for testing."""
    mock_llm = MagicMock()
    async def mock_generate(prompt: str):
        yield "text1"
        yield "text1 text2"
        yield "text1 text2 text3"
    mock_llm.generate.side_effect = mock_generate
    mock_llm.warmup = AsyncMock()
    return MagicMock(return_value=mock_llm), mock_llm

def mock_tts():
    """Mock TTS component for testing."""
    mock_tts = MagicMock()
    async def mock_synthesize(text: str):
        yield b"audio_chunk_1"
        yield b"audio_chunk_2"
        yield b"audio_chunk_3"
        yield b"audio_chunk_4"
        yield b"audio_chunk_5"
        yield b"audio_chunk_6"
    mock_tts.synthesize.side_effect = mock_synthesize
    mock_tts.warmup = AsyncMock()
    return MagicMock(return_value=mock_tts), mock_tts

@pytest.fixture
def mock_gpu_components(monkeypatch):
    """Mock GPU components to avoid dependency issues during tests."""

    mock_tiny_llama_class, mock_tiny_llama_obj = mock_dialogue_core()
    mock_orpheus_class, mock_orpheus_obj = mock_tts()
    
    # create fake modules that contain these symbols
    mock_llm_module = SimpleNamespace(TinyLLama=mock_tiny_llama_class)
    mock_tts_module = SimpleNamespace(Orpheus=mock_orpheus_class)
    
    monkeypatch.setitem(sys.modules, "aichat.components.llm.tiny_llama.model", mock_llm_module)
    monkeypatch.setitem(sys.modules, "aichat.components.tts.orpheus.model", mock_tts_module)
    
    yield {
        "TinyLLama": mock_tiny_llama_obj,
        "Orpheus": mock_orpheus_obj,
    }
    
   