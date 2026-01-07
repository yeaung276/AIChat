"""
Testing Strategy:
- Connection lifecycle management (register, remove)
- State change handling (failed, closed, disconnected)
- Error handling and edge cases
- Resource cleanup and memory leak prevention
"""
import pytest
import uuid
from unittest.mock import AsyncMock, Mock, patch, call

from aichat.rtc.manager import ConnectionManager


class TestConnectionManagerRegister:
    """Test suite for ConnectionManager.register() method."""

    @pytest.mark.asyncio
    async def test_register_creates_new_connection(
        self, mock_rtc_peer_connection, mock_websocket, unique_id
    ):
        """Should create and store RTC connection with WebSocket and Processor."""

        manager = ConnectionManager()

        with patch("aichat.rtc.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection), \
             patch("aichat.rtc.manager.Processor") as mock_processor_cls:
            mock_processor = Mock()
            mock_processor_cls.return_value = mock_processor

            result_id = await manager.register(unique_id, mock_websocket)

            assert result_id == unique_id, "Should return the same connection ID"
            assert unique_id in manager._conns, "Connection should be stored in manager"

            rtc, ws, proc = manager._conns[unique_id]
            assert rtc == mock_rtc_peer_connection, "Should store RTC connection"
            assert ws == mock_websocket, "Should store WebSocket"
            assert proc == mock_processor, "Should store Processor instance"

    @pytest.mark.asyncio
    async def test_register_sets_up_state_change_handler(
        self, mock_rtc_peer_connection, mock_websocket, unique_id
    ):
        """Should register connectionstatechange event handler."""

        manager = ConnectionManager()

        with patch("aichat.rtc.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection), \
             patch("aichat.rtc.manager.Processor"):

            await manager.register(unique_id, mock_websocket)

            assert "connectionstatechange" in mock_rtc_peer_connection._event_handlers, \
                "Should register connectionstatechange handler"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["failed", "closed", "disconnected"])
    async def test_register_handler_removes_connection_on_bad_state(
        self, mock_rtc_peer_connection, mock_websocket, unique_id, state
    ):
        """Should remove connection when state changes to failed/closed/disconnected."""

        manager = ConnectionManager()

        with patch("aichat.rtc.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection), \
             patch("aichat.rtc.manager.Processor"):

            await manager.register(unique_id, mock_websocket)

            # Trigger state change
            mock_rtc_peer_connection.connectionState = state
            handler = mock_rtc_peer_connection._event_handlers["connectionstatechange"]
            await handler()

            assert unique_id not in manager._conns, \
                f"Connection should be removed when state is '{state}'"
            mock_rtc_peer_connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_handler_ignores_good_states(
        self, mock_rtc_peer_connection, mock_websocket, unique_id
    ):
        """Should not remove connection for connecting/connected/new states."""

        manager = ConnectionManager()

        with patch("aichat.rtc.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection), \
             patch("aichat.rtc.manager.Processor"):

            await manager.register(unique_id, mock_websocket)

            # Try good states
            for state in ["new", "connecting", "connected"]:
                mock_rtc_peer_connection.connectionState = state
                handler = mock_rtc_peer_connection._event_handlers["connectionstatechange"]
                await handler()

                assert unique_id in manager._conns, \
                    f"Connection should remain when state is '{state}'"

    @pytest.mark.asyncio
    async def test_register_prevents_recursive_removal(
        self, mock_rtc_peer_connection, mock_websocket, unique_id
    ):
        """Should prevent recursive removal if connection already removed."""

        manager = ConnectionManager()

        with patch("aichat.rtc.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection), \
             patch("aichat.rtc.manager.Processor"):

            await manager.register(unique_id, mock_websocket)

            mock_rtc_peer_connection.connectionState = "failed"
            handler = mock_rtc_peer_connection._event_handlers["connectionstatechange"]

            # Call handler twice (simulating recursive call)
            await handler()
            mock_rtc_peer_connection.close.reset_mock()
            await handler()

            # close should not be called again
            mock_rtc_peer_connection.close.assert_not_called()


class TestConnectionManagerRemove:
    """Test suite for ConnectionManager.remove_rtc() method."""

    @pytest.mark.asyncio
    async def test_remove_rtc_closes_connection_and_deletes(
        self, mock_rtc_peer_connection, mock_websocket, unique_id
    ):
        """Should close RTC connection and remove from storage."""

        manager = ConnectionManager()

        with patch("aichat.rtc.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection), \
             patch("aichat.rtc.manager.Processor"):

            await manager.register(unique_id, mock_websocket)
            assert unique_id in manager._conns

            await manager.remove_rtc(unique_id)

            mock_rtc_peer_connection.close.assert_called_once()
            assert unique_id not in manager._conns, "Connection should be removed from storage"

    @pytest.mark.asyncio
    async def test_remove_rtc_with_nonexistent_id_raises_error(self):
        """Should raise KeyError when removing non-existent connection."""

        manager = ConnectionManager()
        non_existent_id = uuid.uuid4()

        with pytest.raises(KeyError):
            await manager.remove_rtc(non_existent_id)

    @pytest.mark.asyncio
    async def test_remove_rtc_assertion_on_none_connection(self, unique_id):
        """Should raise AssertionError if RTC connection is None"""

        manager = ConnectionManager()
        # Manually inject invalid state
        manager._conns[unique_id] = (None, Mock(), Mock()) # type: ignore

        with pytest.raises(AssertionError, match="connection id not found"):
            await manager.remove_rtc(unique_id)


class TestConnectionManagerAcceptOffer:
    """Test suite for ConnectionManager.accept_offer() method."""

    @pytest.mark.asyncio
    async def test_accept_offer_binds_processor_and_returns_answer(
        self, mock_rtc_peer_connection, mock_websocket, unique_id
    ):
        """Should bind processor, set remote description, create answer, and return SDP."""
  
        manager = ConnectionManager()
        test_sdp = "test_sdp_offer"

        with patch("aichat.rtc.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection), \
             patch("aichat.rtc.manager.Processor") as mock_processor_cls, \
             patch("aichat.rtc.manager.RTCSessionDescription") as mock_session_desc:

            mock_processor = Mock()
            mock_processor.bind = Mock()
            mock_processor_cls.return_value = mock_processor

            await manager.register(unique_id, mock_websocket)

            answer_sdp = await manager.accept_offer(unique_id, test_sdp)

            mock_processor.bind.assert_called_once_with(
                mock_rtc_peer_connection, mock_websocket
            )

            # Verify SDP flow
            mock_session_desc.assert_called_once_with(sdp=test_sdp, type="offer")
            mock_rtc_peer_connection.setRemoteDescription.assert_called_once()
            mock_rtc_peer_connection.createAnswer.assert_called_once()
            mock_rtc_peer_connection.setLocalDescription.assert_called_once()

            assert answer_sdp == "mock_sdp_answer", "Should return local description SDP"

    @pytest.mark.asyncio
    async def test_accept_offer_with_nonexistent_id_raises_error(self):
        """Should raise KeyError when accepting offer for non-existent connection."""

        manager = ConnectionManager()
        non_existent_id = uuid.uuid4()

        with pytest.raises(KeyError):
            await manager.accept_offer(non_existent_id, "test_sdp")

    @pytest.mark.asyncio
    async def test_accept_offer_assertion_on_none_connection(self, unique_id):
        """Should raise AssertionError if RTC connection is None."""

        manager = ConnectionManager()
        manager._conns[unique_id] = (None, Mock(), Mock()) # type: ignore

        with pytest.raises(AssertionError, match="connection id not found"):
            await manager.accept_offer(unique_id, "test_sdp")


class TestConnectionManagerConcurrency:
    """Test suite for concurrent operations."""

    @pytest.mark.asyncio
    async def test_multiple_connections_stored_independently(
        self, mock_websocket
    ):
        """Should handle multiple connections independently."""

        manager = ConnectionManager()
        id1, id2, id3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        with patch("aichat.rtc.manager.RTCPeerConnection") as rtc_cls, \
             patch("aichat.rtc.manager.Processor"):

            # Create mocks with AsyncMock close methods
            rtc1, rtc2, rtc3 = Mock(), Mock(), Mock()
            rtc1.close = AsyncMock()
            rtc2.close = AsyncMock()
            rtc3.close = AsyncMock()
            rtc_cls.side_effect = [rtc1, rtc2, rtc3]

            await manager.register(id1, mock_websocket)
            await manager.register(id2, mock_websocket)
            await manager.register(id3, mock_websocket)

            assert len(manager._conns) == 3
            assert id1 in manager._conns
            assert id2 in manager._conns
            assert id3 in manager._conns

            await manager.remove_rtc(id2)
            assert len(manager._conns) == 2
            assert id1 in manager._conns
            assert id3 in manager._conns
            assert id2 not in manager._conns
