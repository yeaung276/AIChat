"""
Testing Strategy:
- SDP offer/answer negotiation flow
- Message parsing and validation
- Connection lifecycle management
- Error handling and cleanup
- WebSocket communication patterns
"""
import pytest
import json
from unittest.mock import AsyncMock, Mock, patch, call

from aichat.types import MESSAGE_TYPE_SDP_ANSWER, MESSAGE_TYPE_SDP_OFFER


class TestWebSocketEndpoint:
    """Test suite for /ws/sdp WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_accepts_connection(self, mock_websocket):
        """Should accept WebSocket connection on connect."""

        from aichat.rtc.websocket import websocket as ws_handler

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            mock_manager.accept_offer = AsyncMock(return_value="mock_answer")
            mock_manager.remove_rtc = AsyncMock()

            # Setup websocket to end immediately
            async def empty_iter():
                yield

            mock_websocket.iter_text.return_value = empty_iter()

            await ws_handler(mock_websocket)

            mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_handles_sdp_offer(self, mock_websocket):
        """Should handle SDP_OFFER message and return SDP_ANSWER."""

        from aichat.rtc.websocket import websocket as ws_handler

        test_sdp = "test_sdp_offer_content"
        offer_message = json.dumps({
            "type": MESSAGE_TYPE_SDP_OFFER,
            "sdp": test_sdp
        })

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            mock_manager.accept_offer = AsyncMock(return_value="answer_sdp")
            mock_manager.remove_rtc = AsyncMock()

            # Setup websocket to receive one message then stop
            async def mock_iter():
                yield offer_message

            mock_websocket.iter_text = Mock(return_value=mock_iter())

            await ws_handler(mock_websocket)

            # Verify registration happened
            assert mock_manager.register.call_count == 1

            # Verify offer was accepted
            mock_manager.accept_offer.assert_called_once()
            call_args = mock_manager.accept_offer.call_args
            assert call_args[0][1] == test_sdp 

            # Verify answer was sent
            mock_websocket.send_json.assert_called_once()
            sent_message = mock_websocket.send_json.call_args[0][0]
            assert sent_message["type"] == MESSAGE_TYPE_SDP_ANSWER
            assert sent_message["sdp"] == "answer_sdp"

    @pytest.mark.asyncio
    async def test_websocket_registers_with_unique_id(self, mock_websocket):
        """Should register connection with unique UUID."""

        from aichat.rtc.websocket import websocket as ws_handler

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            mock_manager.accept_offer = AsyncMock(return_value="answer")
            mock_manager.remove_rtc = AsyncMock()

            # Setup to process one offer
            offer_message = json.dumps({
                "type": MESSAGE_TYPE_SDP_OFFER,
                "sdp": "test_sdp"
            })

            async def mock_iter():
                yield offer_message

            mock_websocket.iter_text = Mock(return_value=mock_iter())

            await ws_handler(mock_websocket)

            # Check that register was called with a UUID
            assert mock_manager.register.call_count == 1
            conn_id = mock_manager.register.call_args[0][0]
            assert conn_id is not None

            # Check that accept_offer was called with same UUID
            accept_id = mock_manager.accept_offer.call_args[0][0]
            assert accept_id == conn_id


    @pytest.mark.asyncio
    async def test_websocket_ignores_non_offer_messages(self, mock_websocket):
        """Should ignore messages that are not SDP_OFFER."""

        from aichat.rtc.websocket import websocket as ws_handler

        messages = [
            json.dumps({"type": "UNKNOWN_TYPE", "data": "something"}),
            json.dumps({"type": MESSAGE_TYPE_SDP_OFFER, "sdp": "valid_offer"}),
        ]

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            mock_manager.accept_offer = AsyncMock(return_value="answer")
            mock_manager.remove_rtc = AsyncMock()

            async def mock_iter():
                for msg in messages:
                    yield msg

            mock_websocket.iter_text = Mock(return_value=mock_iter())

            await ws_handler(mock_websocket)

            # Should only accept the valid offer
            assert mock_manager.accept_offer.call_count == 1

    @pytest.mark.asyncio
    async def test_websocket_calls_cleanup_on_exit(self, mock_websocket):
        """Should call remove_rtc in finally block."""
        from aichat.rtc.websocket import websocket as ws_handler

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            mock_manager.accept_offer = AsyncMock(return_value="answer")
            mock_manager.remove_rtc = AsyncMock()

            # Empty message stream
            async def empty_iter():
                yield

            mock_websocket.iter_text.return_value = empty_iter()

            await ws_handler(mock_websocket)

            # Cleanup should be called even with no messages
            mock_manager.remove_rtc.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_cleanup_on_exception(self, mock_websocket):
        """Should cleanup connection even when exception occurs."""
        from aichat.rtc.websocket import websocket as ws_handler

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            # Simulate error during accept_offer
            mock_manager.accept_offer = AsyncMock(side_effect=Exception("Test error"))
            mock_manager.remove_rtc = AsyncMock()

            offer_message = json.dumps({
                "type": MESSAGE_TYPE_SDP_OFFER,
                "sdp": "test_sdp"
            })

            async def mock_iter():
                yield offer_message

            mock_websocket.iter_text.return_value = mock_iter()

            await ws_handler(mock_websocket)

            # Cleanup should still be called
            mock_manager.remove_rtc.assert_called_once()


class TestWebSocketMessageParsing:
    """Test suite for message parsing and validation."""

    @pytest.mark.asyncio
    async def test_websocket_handles_malformed_json(self, mock_websocket):
        """Should handle malformed JSON gracefully."""

        from aichat.rtc.websocket import websocket as ws_handler

        malformed_json = "{ this is not valid json }"

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            mock_manager.accept_offer = AsyncMock(return_value="answer")
            mock_manager.remove_rtc = AsyncMock()

            async def mock_iter():
                yield malformed_json

            mock_websocket.iter_text.return_value = mock_iter()

            # Should not raise
            await ws_handler(mock_websocket)

            # accept_offer should not be called
            mock_manager.accept_offer.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_handles_missing_type_field(self, mock_websocket):
        """Should handle messages without 'type' field."""

        from aichat.rtc.websocket import websocket as ws_handler

        message = json.dumps({"sdp": "test_sdp"})  # Missing 'type'

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            mock_manager.accept_offer = AsyncMock(return_value="answer")
            mock_manager.remove_rtc = AsyncMock()

            async def mock_iter():
                yield message

            mock_websocket.iter_text.return_value = mock_iter()

            # Should handle gracefully
            await ws_handler(mock_websocket)

            # Should not process the message
            mock_manager.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_handles_missing_sdp_field(self, mock_websocket):
        """Should handle SDP_OFFER messages without 'sdp' field."""

        from aichat.rtc.websocket import websocket as ws_handler

        message = json.dumps({"type": MESSAGE_TYPE_SDP_OFFER})  # Missing 'sdp'

        with patch("aichat.rtc.websocket.manager") as mock_manager:
            mock_manager.register = AsyncMock()
            mock_manager.accept_offer = AsyncMock(return_value="answer")
            mock_manager.remove_rtc = AsyncMock()

            async def mock_iter():
                yield message

            mock_websocket.iter_text.return_value = mock_iter()

            await ws_handler(mock_websocket)

            # Cleanup should still happen
            mock_manager.remove_rtc.assert_called_once()

