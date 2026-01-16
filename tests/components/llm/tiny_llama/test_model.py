"""
Unit tests for TinyLLama model_cpu module.
Testing Strategy:
- Test method existence and signatures (input/output types)
- No implementation testing (covered in test_implementation.py)
"""
import pytest
import inspect

from aichat.components.llm.tiny_llama.cpu_model import TinyLLamaCPU


class TestTinyLlamaConfigureMethod:
    """Test configure method signature."""

    def test_configure_accepts_correct_parameters(self):
        """Should accept (model: str, temperature: float, max_token: int, device: str)."""
        # Use a real tiny model that exists
        TinyLLamaCPU.configure(
            model="hf-internal-testing/tiny-random-gpt2",
            temperature=0.5,
            max_token=256,
            device="cpu"
        )

        assert TinyLLamaCPU.model is not None
        assert TinyLLamaCPU.tokenizer is not None
        assert TinyLLamaCPU.generation_kwargs is not None

    def test_configure_with_wrong_type_model(self):
        """Should handle wrong type for model parameter."""
        with pytest.raises(Exception):
            TinyLLamaCPU.configure(
                model=123,  # type: ignore Wrong type
                temperature=0.5,
                max_token=256,
                device="cpu"
            )

    def test_configure_with_wrong_type_temperature(self):
        """Should handle wrong type for temperature parameter."""
        with pytest.raises(Exception):
            TinyLLamaCPU.configure(
                model="test-model",
                temperature="wrong",  # type: ignore Wrong type
                max_token=256,
                device="cpu"
            )

    def test_configure_with_wrong_type_max_token(self):
        """Should handle wrong type for max_token parameter."""
        with pytest.raises(Exception):
            TinyLLamaCPU.configure(
                model="test-model",
                temperature=0.5,
                max_token="wrong",  # type: ignore Wrong type
                device="cpu"
            )


class TestTinyLlamaGenerateMethod:
    """Test generate method signature."""

    @pytest.mark.asyncio
    async def test_generate_accepts_string_returns_async_generator(self):
        """Should accept (text: str) and return AsyncGenerator[str, None]."""
        instance = TinyLLamaCPU()

        try:
            result = instance.generate("test input")
            assert inspect.isasyncgen(result)
            await result.aclose()
        except Exception as e:
            # Should fail because model not configured, not signature issue
            assert "not configured" in str(e).lower() or "nonetype" in str(e).lower()

    @pytest.mark.asyncio
    async def test_generate_with_wrong_type(self):
        """Should handle wrong type for text parameter."""
        TinyLLamaCPU.configure(
            model="hf-internal-testing/tiny-random-gpt2",
            temperature=0.5,
            max_token=10,
            device="cpu"
        )
        instance = TinyLLamaCPU()

        with pytest.raises(ValueError) as exc_info:
            result = instance.generate(123)  # type: ignore Wrong type
            async for _ in result:
                pass

        assert "text input must be of type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_when_not_configured(self):
        """Should raise exception when model not configured."""
        instance = TinyLLamaCPU()
        instance.model = None
        instance.tokenizer = None

        with pytest.raises(Exception) as exc_info:
            result = instance.generate("test")
            if inspect.isasyncgen(result):
                async for _ in result:
                    pass

        assert "not configured" in str(exc_info.value).lower()
