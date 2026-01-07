"""
Testing Strategy:
- Test full end-to-end workflows with real Dummy implementations
- Verify complete data flow: audio -> STT -> LLM queue -> LLM -> WebSocket
- Test concurrent audio and video processing
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

from aichat.pipeline.processor import Processor
from aichat.types import MESSAGE_TYPE_SPEECH_DEBUG, MESSAGE_TYPE_SPEECH_SPEAK


class TestProcessorIntegration:
    """Full integration tests with Processor + real components."""

    def test_processor_initializes_with_dummy_models_by_default(self):
        """Should initialize with models"""
        
        processor = Processor()

        # Assert
        assert processor.stt is not None
        assert processor.llm is not None
        assert processor.video_analyzer is not None
        assert processor.llm_queue is not None
        assert isinstance(processor.llm_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_full_audio_to_llm_to_websocket_flow(self, mock_websocket, mock_rtc_peer_connection, mock_audio_frame, mock_audio_track):
        """Should process: audio -> STT -> LLM queue -> LLM -> WebSocket."""
        # Arrange
        processor = Processor()

        # Override STT to return a message immediately
        processor.stt.accept = AsyncMock(return_value="user said hello")

        # Setup processor with WebSocket
        processor.bind(mock_rtc_peer_connection, mock_websocket)

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

            # Should have sent both debug and speak messages
            assert mock_websocket.send_json.call_count >= 2

            # Verify debug message was sent
            calls = mock_websocket.send_json.call_args_list
            debug_messages = [c[0][0] for c in calls if c[0][0]["type"] == MESSAGE_TYPE_SPEECH_DEBUG]
            assert len(debug_messages) >= 1
            assert debug_messages[0]["data"]["message"] == "user said hello"

            # Verify LLM response was sent
            speak_messages = [c[0][0] for c in calls if c[0][0]["type"] == MESSAGE_TYPE_SPEECH_SPEAK]
            assert len(speak_messages) >= 1
            assert speak_messages[0]["data"] == "Hi, I am Aura. What can I help you with?"

            # Cleanup
            if processor.audio_task:
                processor.audio_task.cancel()
                try:
                    await processor.audio_task
                except asyncio.CancelledError:
                    pass

            if processor.llm_task:
                processor.llm_task.cancel()
                try:
                    await processor.llm_task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_video_and_audio_tracks_work_concurrently(self, mock_websocket, mock_rtc_peer_connection, mock_audio_track, mock_video_track):
        """Should handle video and audio tracks simultaneously."""
        # Arrange
        processor = Processor()
        processor.stt.accept = AsyncMock(return_value=None)
        processor.video_analyzer.accept = AsyncMock()

        processor.bind(mock_rtc_peer_connection, mock_websocket)

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
            for task in [processor.audio_task, processor.video_task, processor.llm_task]:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    

