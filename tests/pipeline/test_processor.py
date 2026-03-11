"""
Testing Strategy:
- Test Processor's orchestration logic with mocked Memory
- Verify correct data flow through audio/video/LLM pipeline
- Task lifecycle and cleanup
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from aichat.pipeline.processor import Processor, ProfiledResult
from aichat.types import MESSAGE_TYPE_AVATAR_SPEAK


class TestProcessorInitialization:
    """Initialization tests with required dependencies."""

    def test_processor_initializes_with_memory(self):
        """Should initialize with provided models and memory."""
        mock_memory = Mock()

        processor = Processor(
            speech="dummy",
            video="dummy",
            llm="dummy",
            tts="dummy",
            voice="test_voice",
            context=mock_memory
        )

        assert processor.stt is not None
        assert processor.llm is not None
        assert processor.video_analyzer is not None
        assert processor.llm_queue is not None
        assert isinstance(processor.llm_queue, asyncio.Queue)
        assert processor.context == mock_memory

    def test_processor_tasks_initialize_as_none(self):
        """Should have no running tasks before bind()."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )

        assert processor.video_task is None
        assert processor.audio_task is None
        assert processor.llm_task is None
        assert processor.tts_task is None

    @pytest.mark.asyncio
    async def test_processor_bind_stores_websocket(self, mock_websocket, mock_rtc_peer_connection):
        """Should store WebSocket reference after bind()."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )

        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        assert processor.ws == mock_websocket

        # Cleanup
        await processor.close()

    @pytest.mark.asyncio
    async def test_processor_bind_registers_track_handler(self, mock_rtc_peer_connection, mock_websocket):
        """Should register 'track' event handler on RTC connection."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )

        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        assert "track" in mock_rtc_peer_connection._event_handlers

        # Cleanup
        await processor.close()

    @pytest.mark.asyncio
    async def test_processor_bind_creates_llm_task(self, mock_rtc_peer_connection, mock_websocket):
        """Should create LLM processing task."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )

        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        assert processor.llm_task is not None
        assert isinstance(processor.llm_task, asyncio.Task)

        await processor.close()

    @pytest.mark.asyncio
    async def test_processor_bind_creates_tts_task(self, mock_rtc_peer_connection, mock_websocket):
        """Should create TTS processing task."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )

        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        assert processor.tts_task is not None
        assert isinstance(processor.tts_task, asyncio.Task)

        await processor.close()

    @pytest.mark.asyncio
    async def test_processor_creates_audio_task_on_audio_track(self, mock_websocket, mock_rtc_peer_connection):
        """Should create audio processing task when audio track is received."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )
        processor.stt.accept = AsyncMock(return_value=None)

        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        # Mock audio track
        mock_audio_track = AsyncMock()
        mock_audio_track.kind = "audio"
        mock_audio_track.recv = AsyncMock(side_effect=asyncio.CancelledError())

        track_handler = mock_rtc_peer_connection._event_handlers["track"]

        with patch("aichat.pipeline.processor.AudioResampler"):
            await track_handler(mock_audio_track)
            await asyncio.sleep(0.05)

            # Assert audio task was created
            assert processor.audio_task is not None
            assert isinstance(processor.audio_task, asyncio.Task)

        await processor.close()

    @pytest.mark.asyncio
    async def test_processor_creates_video_task_on_video_track(self, mock_websocket, mock_rtc_peer_connection):
        """Should create video processing task when video track is received."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )
        processor.video_analyzer.accept = AsyncMock()

        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        # Mock video track
        mock_video_track = AsyncMock()
        mock_video_track.kind = "video"
        mock_video_track.recv = AsyncMock(side_effect=asyncio.CancelledError())

        track_handler = mock_rtc_peer_connection._event_handlers["track"]
        await track_handler(mock_video_track)
        await asyncio.sleep(0.05)

        # Assert video task was created
        assert processor.video_task is not None
        assert isinstance(processor.video_task, asyncio.Task)

        await processor.close()


class TestProcessorAudioPipeline:
    """Test audio processing pipeline with mocked components."""

    @pytest.mark.asyncio
    async def test_audio_track_triggers_stt_processing(self, mock_audio_frame, mock_audio_track):
        """Should process audio frames through STT."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )

        # Mock STT to return a message
        processor.stt.accept = AsyncMock(return_value="test transcription")

        # Mock audio track with mock frame
        mock_audio_track.recv.side_effect = [mock_audio_frame, asyncio.CancelledError()]

        with patch("aichat.pipeline.processor.AudioResampler") as mock_resampler_cls:
            mock_resampler = Mock()
            mock_resampler.resample = Mock(return_value=[mock_audio_frame])
            mock_resampler_cls.return_value = mock_resampler

            with pytest.raises(asyncio.CancelledError):
                await processor._read_audio_track(mock_audio_track)

            assert processor.stt.accept.call_count == 1
            assert not processor.llm_queue.empty()
            message = await processor.llm_queue.get()
            assert message.incoming == "test transcription"

    @pytest.mark.asyncio
    async def test_audio_track_ignores_none_from_stt(self, mock_audio_frame, mock_audio_track):
        """Should not queue None messages from STT."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )

        # Mock stt to return none
        processor.stt.accept = AsyncMock(return_value=None)

        # Mock audio track with mock frame
        mock_audio_track.recv.side_effect = [mock_audio_frame, asyncio.CancelledError()]

        with patch("aichat.pipeline.processor.AudioResampler") as mock_resampler_cls:
            mock_resampler = Mock()
            mock_resampler.resample = Mock(return_value=[mock_audio_frame])
            mock_resampler_cls.return_value = mock_resampler

            with pytest.raises(asyncio.CancelledError):
                await processor._read_audio_track(mock_audio_track)

            assert processor.llm_queue.empty()


class TestProcessorVideoPipeline:
    """Test video processing pipeline."""

    @pytest.mark.asyncio
    async def test_video_track_sends_frames_to_analyzer(self, mock_video_frame, mock_video_track):
        """Should send video frames to analyzer."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )
        processor.video_analyzer.accept = AsyncMock()

        # Mock video track with mock frame
        mock_video_track.recv.side_effect = [mock_video_frame, asyncio.CancelledError()]

        with pytest.raises(asyncio.CancelledError):
            await processor._read_video_track(mock_video_track)

        processor.video_analyzer.accept.assert_called_once_with(mock_video_frame)


class TestProcessorLLMPipeline:
    """Test LLM queue processing with Memory integration."""

    @pytest.mark.asyncio
    async def test_llm_queue_processes_message_with_memory(self, mock_websocket):
        """Should add message to memory, generate response, and save to memory."""
        mock_memory = AsyncMock()
        mock_memory.get_context = AsyncMock(return_value="test input")
        mock_memory.add = AsyncMock()

        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )
        processor.ws = mock_websocket

        await processor.llm_queue.put(ProfiledResult(incoming="test input", profiled=[{"component": "stt_out", "time": 1.0}]))

        task1 = asyncio.create_task(processor._read_llm_queue(processor.llm_queue))
        task2 = asyncio.create_task(processor._read_tts_queue(processor.tts_queue))
        await asyncio.sleep(0.1)
        task1.cancel()

        try:
            await task1
        except asyncio.CancelledError:
            pass
        
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass

        # Should add user message to memory
        mock_memory.add.assert_any_call(actor="user", message="test input")

        # Should get context from memory
        mock_memory.get_context.assert_called_once()

        # Should send TTS audio via WebSocket
        assert mock_websocket.send_json.call_count >= 1
        speak_call = mock_websocket.send_json.call_args_list[0]
        speak_msg = speak_call[0][0]
        assert speak_msg["type"] == MESSAGE_TYPE_AVATAR_SPEAK
        assert "audio" in speak_msg["data"]
        assert "meta" in speak_msg["data"]

        # Should add assistant response to memory
        assert mock_memory.add.call_count >= 2

    @pytest.mark.asyncio
    async def test_llm_queue_processes_multiple_messages(self, mock_websocket):
        """Should process messages from queue in FIFO order."""
        mock_memory = AsyncMock()
        mock_memory.get_context = AsyncMock(return_value="context")
        mock_memory.add = AsyncMock()

        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )
        processor.ws = mock_websocket

        await processor.llm_queue.put(ProfiledResult(incoming="first", profiled=[]))
        await processor.llm_queue.put(ProfiledResult(incoming="second", profiled=[]))

        task = asyncio.create_task(processor._read_llm_queue(processor.llm_queue))
        await asyncio.sleep(0.3)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should process both messages
        add_calls = mock_memory.add.call_args_list
        user_messages = [call[1]["message"] for call in add_calls if call[1]["actor"] == "user"]

        assert len(user_messages) >= 2
        assert user_messages[0] == "first"
        assert user_messages[1] == "second"


class TestProcessorTTSPipeline:
    """Test TTS queue processing with audio synthesis."""

    @pytest.mark.asyncio
    async def test_tts_queue_synthesizes_and_sends_audio(self, mock_websocket):
        """Should synthesize audio from TTS queue and send via WebSocket."""
        mock_memory = AsyncMock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )
        processor.ws = mock_websocket

        # Put message in TTS queue
        await processor.tts_queue.put(ProfiledResult(response="Hello world", profiled=[{"component": "stt_out", "time": 1.0}]))

        task = asyncio.create_task(processor._read_tts_queue(processor.tts_queue))
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should send synthesized audio via WebSocket
        assert mock_websocket.send_json.call_count >= 1
        call = mock_websocket.send_json.call_args_list[0]
        message = call[0][0]
        assert message["type"] == MESSAGE_TYPE_AVATAR_SPEAK
        assert "audio" in message["data"]
        assert "meta" in message["data"]


class TestProcessorCleanup:
    """Test task cleanup."""

    @pytest.mark.asyncio
    async def test_close_cancels_all_tasks(self, mock_websocket, mock_rtc_peer_connection):
        """Should cancel all running tasks."""
        mock_memory = Mock()
        processor = Processor(
            speech="dummy", video="dummy", llm="dummy", tts="dummy",
            voice="test_voice", context=mock_memory
        )
        processor.stt.accept = AsyncMock(return_value=None)
        processor.video_analyzer.accept = AsyncMock()

        await processor.bind(mock_rtc_peer_connection, mock_websocket)

        # Create audio and video tasks
        mock_audio_track = AsyncMock()
        mock_audio_track.kind = "audio"
        mock_audio_track.recv = AsyncMock(side_effect=[AsyncMock(), asyncio.CancelledError()])

        mock_video_track = AsyncMock()
        mock_video_track.kind = "video"
        mock_video_track.recv = AsyncMock(side_effect=[AsyncMock(), asyncio.CancelledError()])

        track_handler = mock_rtc_peer_connection._event_handlers["track"]

        with patch("aichat.pipeline.processor.AudioResampler"):
            await track_handler(mock_audio_track)
            await track_handler(mock_video_track)
            await asyncio.sleep(0.05)

        # All tasks should be created
        assert processor.audio_task is not None
        assert processor.video_task is not None
        assert processor.llm_task is not None

        # Close should cancel all tasks
        await processor.close()

        # Give tasks time to be cancelled
        await asyncio.sleep(0.05)

        assert processor.audio_task.cancelled()
        assert processor.video_task.cancelled()
        assert processor.llm_task.cancelled()
