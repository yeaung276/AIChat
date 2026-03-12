"""
Tests for aichat.pipeline.context.Context

Unit tests: mock heavy ML deps (spacy, transformers, DB) for fast isolated logic.
Behavioral tests: real models, test trigger conditions and state updates.
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool

import aichat.pipeline.context as ctx_module
from aichat.db_models.chat import Character
from aichat.db_models.user import User
from aichat.pipeline.context import Context
from aichat.types import MESSAGE_TYPE_TRANSCRIPT


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

class FakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, data: dict):
        self.sent.append(data)


def _make_chat(transcripts=None, prompt="Initial situation"):
    chat = MagicMock()
    chat.id = 1
    chat.prompt = prompt
    chat.transcripts = list(transcripts or [])
    return chat


def _make_ctx(transcripts=None, prompt="Initial situation"):
    """Fast ctx — __init__ no longer loads models (they are module-level)."""
    chat = _make_chat(transcripts=transcripts, prompt=prompt)
    ws = FakeWebSocket()
    ctx = Context(prompt=chat.prompt, ws=ws)
    if transcripts:
        ctx.messages = list(transcripts)
    return ctx


# ---------------------------------------------------------------------------
# _get_window  (pure logic — no mocks needed)
# ---------------------------------------------------------------------------

class TestGetWindow:
    def test_empty_messages_returns_none(self):
        ctx = _make_ctx()
        ctx.messages = []
        assert ctx._get_window() is None

    def test_below_min_words_returns_none(self):
        ctx = _make_ctx()
        ctx.messages = [
            {"actor": "user", "message": "hello"},
            {"actor": "assistant", "message": "hi"},
        ]
        assert ctx._get_window(min_words=50) is None

    def test_exactly_at_min_words_returns_window(self):
        ctx = _make_ctx()
        msg = {"actor": "user", "message": " ".join(["word"] * 50)}
        ctx.messages = [msg]
        assert ctx._get_window(min_words=50) == [msg]

    def test_returns_only_recent_messages_needed(self):
        ctx = _make_ctx()
        old = {"actor": "user", "message": "old message here"}
        r1 = {"actor": "user", "message": " ".join(["word"] * 30)}
        r2 = {"actor": "assistant", "message": " ".join(["word"] * 25)}
        ctx.messages = [old, r1, r2]
        window = ctx._get_window(min_words=50)
        assert old not in window
        assert r1 in window and r2 in window

    def test_returns_all_messages_when_all_needed(self):
        ctx = _make_ctx()
        msgs = [{"actor": "user", "message": " ".join(["word"] * 20)}] * 3
        ctx.messages = msgs
        assert ctx._get_window(min_words=50) == msgs


# ---------------------------------------------------------------------------
# add  (unit — only WS is a stub)
# ---------------------------------------------------------------------------

class TestAdd:
    @pytest.mark.asyncio
    async def test_message_appended(self):
        ctx = _make_ctx()
        ctx.messages = []
        ctx._pending = False
        with patch.object(ctx, "_get_window", return_value=None):
            await ctx.add("user", "hello")
        assert {"actor": "user", "message": "hello"} in ctx.messages

    @pytest.mark.asyncio
    async def test_sets_pending_and_schedules_housekeeping_when_window_exists(self):
        ctx = _make_ctx()
        ctx._pending = False
        window = [{"actor": "user", "message": " ".join(["word"] * 55)}]

        with patch.object(ctx, "_get_window", return_value=window), \
             patch.object(asyncio, "get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = MagicMock(return_value=asyncio.Future())
            await ctx.add("user", "trigger")

        assert ctx._pending is True
        mock_loop.return_value.run_in_executor.assert_called_once_with(
            ctx._executor, ctx._update_topic, window
        )

    @pytest.mark.asyncio
    async def test_does_not_trigger_when_already_pending(self):
        ctx = _make_ctx()
        ctx._pending = True
        window = [{"actor": "user", "message": " ".join(["word"] * 55)}]

        with patch.object(ctx, "_get_window", return_value=window), \
             patch.object(asyncio, "get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = MagicMock()
            await ctx.add("user", "no trigger")

        mock_loop.return_value.run_in_executor.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_trigger_when_no_window(self):
        ctx = _make_ctx()
        ctx._pending = False

        with patch.object(ctx, "_get_window", return_value=None), \
             patch.object(asyncio, "get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = MagicMock()
            await ctx.add("user", "not enough")

        mock_loop.return_value.run_in_executor.assert_not_called()


# ---------------------------------------------------------------------------
# _update_topic  (unit — mock model.generate + tokenizer)
# ---------------------------------------------------------------------------

class TestHousekeepingUnit:
    def _setup(self, ctx, decoded_output="User is discussing something. Extra line."):
        mock_inputs = MagicMock()
        mock_inputs["input_ids"].shape = (1, 10)
        mock_inputs.to = MagicMock(return_value=mock_inputs)

        mock_tok = MagicMock()
        mock_tok.return_value = mock_inputs
        mock_tok.decode = MagicMock(return_value=decoded_output)
        mock_tok.eos_token_id = 0

        mock_mod = MagicMock()
        mock_mod.generate = MagicMock(return_value=[MagicMock()])
        mock_mod.device = "cpu"

        ctx._mock_tokenizer = mock_tok
        ctx._mock_model = mock_mod

    def test_prompt_updated_to_first_line_of_output(self):
        ctx = _make_ctx(prompt="old prompt")
        self._setup(ctx, "New situation here.\nIgnore this line.")
        ctx.messages = [{"actor": "user", "message": "hello world"}]

        with patch("aichat.pipeline.context.tokenizer", ctx._mock_tokenizer), \
             patch("aichat.pipeline.context.model", ctx._mock_model):
            ctx._update_topic(ctx.messages)

        assert ctx.prompt == "New situation here."

    def test_pending_reset_to_false(self):
        ctx = _make_ctx()
        self._setup(ctx)
        ctx._pending = True
        ctx.messages = []

        with patch("aichat.pipeline.context.tokenizer", ctx._mock_tokenizer), \
             patch("aichat.pipeline.context.model", ctx._mock_model):
            ctx._update_topic([])

        assert ctx._pending is False


# ---------------------------------------------------------------------------
# get_context  (unit — real build_prompt, only ctx deps mocked)
# ---------------------------------------------------------------------------

class TestGetContext:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        ctx = _make_ctx(transcripts=[{"actor": "user", "message": "hello"}], prompt="p")
        result = await ctx.get_context(emotion="happy")
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.asyncio
    async def test_contains_situation(self):
        ctx = _make_ctx(transcripts=[{"actor": "user", "message": "hello"}], prompt="User is curious")
        assert "User is curious" in await ctx.get_context(emotion="neutral")

    @pytest.mark.asyncio
    async def test_contains_emotion(self):
        ctx = _make_ctx(transcripts=[{"actor": "user", "message": "hello"}])
        assert "excited" in await ctx.get_context(emotion="excited")

    @pytest.mark.asyncio
    async def test_uses_last_message(self):
        msgs = [
            {"actor": "user", "message": "first"},
            {"actor": "assistant", "message": "response"},
            {"actor": "user", "message": "final question"},
        ]
        ctx = _make_ctx(transcripts=msgs)
        assert "final question" in await ctx.get_context(emotion="neutral")


# ---------------------------------------------------------------------------
# Behavioral tests — real spacy + real Qwen, no ML mocks
# ---------------------------------------------------------------------------

@pytest.mark.skip()
class TestBehavioral:
    """End-to-end behavior with real ML models."""

    def test_housekeeping_updates_prompt(self, real_ctx):
        old_prompt = real_ctx.prompt
        window = [
            {"actor": "user", "message": "I just won two hundred dollars on a lottery ticket today!"},
            {"actor": "assistant", "message": "That is wonderful, congratulations on your big win!"},
        ]
        real_ctx.messages = list(window)
        real_ctx._pending = True
        real_ctx._update_topic(window)
        assert isinstance(real_ctx.prompt, str)
        assert len(real_ctx.prompt) > 0
        assert "\n" not in real_ctx.prompt

    @pytest.mark.asyncio
    async def test_add_triggers_housekeeping_at_word_limit(self, real_ctx):
        real_ctx.ws = FakeWebSocket()
        real_ctx._pending = False
        real_ctx.messages = [{"actor": "user", "message": " ".join(["word"] * 49)}]
        await real_ctx.add("user", "one more word here to push over the limit")
        assert real_ctx._pending is True

    @pytest.mark.asyncio
    async def test_add_does_not_trigger_below_word_limit(self, real_ctx):
        real_ctx.ws = FakeWebSocket()
        real_ctx._pending = False
        real_ctx.messages = []
        await real_ctx.add("user", "just a few words")
        assert real_ctx._pending is False
