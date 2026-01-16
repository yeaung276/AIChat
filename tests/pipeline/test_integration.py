"""
Testing Strategy:
- Test full end-to-end workflows with real components
- Verify complete data flow: audio -> STT -> LLM queue -> LLM -> Memory -> WebSocket
- Use real Memory, real LLM, real components - only mock infrastructure
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from sqlmodel import Session, create_engine, SQLModel
from sqlalchemy.pool import StaticPool

from aichat.pipeline.processor import Processor
from aichat.pipeline.memory import Memory
from aichat.types import MESSAGE_TYPE_AVATAR_SPEAK, MESSAGE_TYPE_TRANSCRIPT
from aichat.db_models.chat import Chat
from aichat.db_models.user import User


@pytest.fixture
def test_db():
    """Create in-memory database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def test_chat(test_db):
    """Create test chat in database."""
    user = User(username="testuser", pwd_hash="hashed_pwd", screen_name="Test User", bio=None)
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

    assert user.id is not None

    chat = Chat(
        user_id=user.id,
        name="Test Chat",
        voice="test_voice",
        face="test_face",
        prompt="You are helpful",
        llm="dummy",
        transcripts=[]
    )
    test_db.add(chat)
    test_db.commit()
    test_db.refresh(chat)
    return chat


class TestProcessorIntegration:
    """Full integration tests with Processor + real components."""

    @pytest.mark.asyncio
    async def test_full_audio_to_llm_to_websocket_flow(self, mock_websocket, mock_rtc_peer_connection, mock_audio_frame, mock_audio_track, test_db, test_chat):
        """Should process: audio -> STT -> LLM queue -> LLM -> Memory -> WebSocket."""

        mem = Memory(chat=test_chat, db=test_db, ws=mock_websocket)

        processor = Processor(
            speech="dummy",
            video="dummy",
            llm="dummy",
            voice="test_voice",
            memory=mem
        )

        # Override STT to return a message immediately
        processor.stt.accept = AsyncMock(return_value="user said hello")

        # Setup processor with WebSocket
        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        # Return frame once then cancel
        mock_audio_track.recv.side_effect = [mock_audio_frame, asyncio.CancelledError()]

        # Act - Simulate receiving audio track
        with patch("aichat.pipeline.processor.AudioResampler") as mock_resampler_cls:
            mock_resampler = Mock()
            mock_resampler.resample = Mock(return_value=[mock_audio_frame])
            mock_resampler_cls.return_value = mock_resampler

            # Trigger track handler
            track_handler = mock_rtc_peer_connection._event_handlers["track"]
            await track_handler(mock_audio_track)

            # Wait for audio processing
            await asyncio.sleep(0.1)

            # Wait for LLM processing
            await asyncio.sleep(0.2)

            # Verify Memory was used and data persisted
            assert len(mem.messages) == 2  # user + assistant
            assert mem.messages[0]["actor"] == "user"
            assert mem.messages[0]["message"] == "user said hello"
            assert mem.messages[1]["actor"] == "assistant"

            # Verify database persistence
            test_db.refresh(test_chat)
            assert len(test_chat.transcripts) == 2

            # Verify WebSocket sent messages (transcripts + avatar speak)
            assert mock_websocket.send_json.call_count >= 2
            calls = mock_websocket.send_json.call_args_list

            # Check transcript messages
            transcript_messages = [c[0][0] for c in calls if c[0][0]["type"] == MESSAGE_TYPE_TRANSCRIPT]
            assert len(transcript_messages) == 2

            # Check avatar speak messages
            speak_messages = [c[0][0] for c in calls if c[0][0]["type"] == MESSAGE_TYPE_AVATAR_SPEAK]
            assert len(speak_messages) >= 1
            assert "text" in speak_messages[0]["data"]

        # Cleanup
        await processor.close()

    @pytest.mark.asyncio
    async def test_video_and_audio_tracks_work_concurrently(self, mock_websocket, mock_rtc_peer_connection, mock_audio_track, mock_video_track, test_db, test_chat):
        """Should handle video and audio tracks simultaneously."""
        # Use real Memory instead of mock
        real_memory = Memory(chat=test_chat, db=test_db, ws=mock_websocket)

        processor = Processor(
            speech="dummy",
            video="dummy",
            llm="dummy",
            voice="test_voice",
            memory=real_memory
        )
        processor.stt.accept = AsyncMock(return_value=None)
        processor.video_analyzer.accept = AsyncMock()

        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        # Create tracks
        mock_audio_track.recv = AsyncMock(side_effect=asyncio.CancelledError())
        mock_video_track.recv = AsyncMock(side_effect=asyncio.CancelledError())

        with patch("aichat.pipeline.processor.AudioResampler"):
            await mock_rtc_peer_connection._event_handlers["track"](mock_audio_track)
            await mock_rtc_peer_connection._event_handlers["track"](mock_video_track)

            await asyncio.sleep(0.1)

            assert processor.audio_task is not None
            assert processor.video_task is not None

        # Cleanup
        await processor.close()
