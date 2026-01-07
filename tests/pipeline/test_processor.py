"""
Testing Strategy:
- Integration testing with real Dummy components (default behavior)
- Test Processor's orchestration logic, not factory logic
- Verify correct data flow through audio/video/LLM pipeline
- Task lifecycle and cleanup
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

from aichat.pipeline.processor import Processor
from aichat.types import MESSAGE_TYPE_SPEECH_DEBUG, MESSAGE_TYPE_SPEECH_SPEAK


class TestProcessorInitialization:
    """Initialization tests using real Dummy components."""

    def test_processor_initializes_with_dummy_models_by_default(self):
        """Should initialize with Dummy models when factory not configured."""

        processor = Processor()

        assert processor.stt is not None
        assert processor.llm is not None
        assert processor.video_analyzer is not None
        assert processor.llm_queue is not None
        assert isinstance(processor.llm_queue, asyncio.Queue)

    def test_processor_tasks_initialize_as_none(self):
        """Should have no running tasks before bind()."""
        processor = Processor()

        assert processor.video_task is None
        assert processor.audio_task is None
        assert processor.llm_task is None

    @pytest.mark.asyncio
    async def test_processor_bind_stores_websocket(self, mock_websocket, mock_rtc_peer_connection):
        """Should store WebSocket reference after bind()."""
        processor = Processor()

        processor.bind(mock_rtc_peer_connection, mock_websocket)

        assert processor.websocket == mock_websocket

    @pytest.mark.asyncio
    async def test_processor_bind_registers_track_handler(self, mock_rtc_peer_connection, mock_websocket):
        """Should register 'track' event handler on RTC connection."""
        processor = Processor()

        processor.bind(mock_rtc_peer_connection, mock_websocket)

        assert "track" in mock_rtc_peer_connection._event_handlers

    @pytest.mark.asyncio
    async def test_processor_bind_creates_llm_task(self, mock_rtc_peer_connection, mock_websocket):
        """Should create LLM processing task."""

        processor = Processor()

        processor.bind(mock_rtc_peer_connection, mock_websocket)

        assert processor.llm_task is not None
        assert isinstance(processor.llm_task, asyncio.Task)

        processor.llm_task.cancel()
        try:
            await processor.llm_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_processor_creates_audio_task_on_audio_track(self, mock_websocket, mock_rtc_peer_connection):
        """Should create audio processing task when audio track is received."""
        processor = Processor()
        processor.stt.accept = AsyncMock(return_value=None)


        processor.bind(mock_rtc_peer_connection, mock_websocket)

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

            for task in [processor.audio_task, processor.llm_task]:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_processor_creates_video_task_on_video_track(self, mock_websocket, mock_rtc_peer_connection):
        """Should create video processing task when video track is received."""
        processor = Processor()
        processor.video_analyzer.accept = AsyncMock()

        processor.bind(mock_rtc_peer_connection, mock_websocket)

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

        for task in [processor.video_task, processor.llm_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


class TestProcessorAudioPipeline:
    """Test audio processing pipeline with mocked components."""

    @pytest.mark.asyncio
    async def test_audio_track_triggers_stt_processing(self, mock_audio_frame, mock_audio_track):
        """Should process audio frames through STT."""

        processor = Processor()

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
            assert message == "test transcription"

    @pytest.mark.asyncio
    async def test_audio_track_ignores_none_from_stt(self, mock_audio_frame, mock_audio_track):
        """Should not queue None messages from STT."""

        processor = Processor()
        
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

        processor = Processor()
        processor.video_analyzer.accept = AsyncMock()

        # Mock video track with mock frame
        mock_video_track.recv.side_effect = [mock_video_frame, asyncio.CancelledError()]

        with pytest.raises(asyncio.CancelledError):
            await processor._read_video_track(mock_video_track)

        processor.video_analyzer.accept.assert_called_once_with(mock_video_frame)


class TestProcessorLLMPipeline:
    """Test LLM queue processing."""

    @pytest.mark.asyncio
    async def test_llm_queue_sends_debug_message(self, mock_websocket):
        """Should send debug message."""

        processor = Processor()
        processor.websocket = mock_websocket

        await processor.llm_queue.put("test input")

        task = asyncio.create_task(processor._read_llm_queue(processor.llm_queue))
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Assert
        assert mock_websocket.send_json.call_count >= 1
        debug_call = mock_websocket.send_json.call_args_list[0]
        debug_msg = debug_call[0][0]
        assert debug_msg["type"] == MESSAGE_TYPE_SPEECH_DEBUG
        assert debug_msg["data"]["actor"] == "user"
        assert debug_msg["data"]["message"] == "test input"

    @pytest.mark.asyncio
    async def test_llm_queue_generates_and_sends_response(self, mock_websocket):
        """Should generate LLM response and send as SPEECH_SPEAK."""

        processor = Processor()
        processor.websocket = mock_websocket

        await processor.llm_queue.put("hello")

        task = asyncio.create_task(processor._read_llm_queue(processor.llm_queue))
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Assert
        assert mock_websocket.send_json.call_count >= 2

        # Check SPEECH_SPEAK message
        speak_call = mock_websocket.send_json.call_args_list[-1]
        speak_msg = speak_call[0][0]
        assert speak_msg["type"] == MESSAGE_TYPE_SPEECH_SPEAK
        # DummyLLM returns "Hi, I am Aura. What can I help you with?"
        assert speak_msg["data"] == "Hi, I am Aura. What can I help you with?"

    @pytest.mark.asyncio
    async def test_llm_queue_processes_multiple_messages_in_order(self, mock_websocket):
        """Should process messages from queue in FIFO order."""
 
        processor = Processor()
        processor.websocket = mock_websocket

        await processor.llm_queue.put("first")
        await processor.llm_queue.put("second")

        task = asyncio.create_task(processor._read_llm_queue(processor.llm_queue))
        await asyncio.sleep(0.3)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert mock_websocket.send_json.call_count >= 4  # 2 debug + 2 speak messages

        # Check order of debug messages
        calls = mock_websocket.send_json.call_args_list
        debug_messages = [c[0][0] for c in calls if c[0][0]["type"] == MESSAGE_TYPE_SPEECH_DEBUG]

        assert len(debug_messages) >= 2
        assert debug_messages[0]["data"]["message"] == "first"
        assert debug_messages[1]["data"]["message"] == "second"
