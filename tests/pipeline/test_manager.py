"""
Testing Strategy:
- Test critical ConnectionManager execution paths
- Test registration/deregistration lifecycle
- Test RTC state change handling (connected, failed, closed)
- Verify proper binding of processor and memory
- Verify WebSocket avatar initialization on connect
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch, call
from aiortc import RTCSessionDescription

from aichat.pipeline.manager import ConnectionManager
from aichat.db_models.chat import Chat
from aichat.types import MESSAGE_TYPE_AVATAR_INITIALIZE


class TestConnectionManager:
    """Test critical ConnectionManager flows."""

    @pytest.fixture
    def manager(self):
        """Create ConnectionManager instance."""
        return ConnectionManager()

    @pytest.fixture
    def mock_chat(self):
        """Create mock Chat object."""
        chat = Mock(spec=Chat)
        chat.id = 123
        chat.voice = "test_voice"
        chat.face = "test_face"
        chat.llm = "dummy"
        chat.prompt = "You are a helpful assistant"
        chat.transcripts = []
        return chat

    @pytest.mark.asyncio
    async def test_register_creates_connection_with_all_components(
        self, manager, mock_chat, mock_websocket, mock_rtc_peer_connection, test_db
    ):
        """Should create RTC, Processor, and Memory, then bind them together."""
        sdp_offer = "mock_sdp_offer"

        with patch("aichat.pipeline.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection):
            with patch("aichat.pipeline.manager.Processor") as mock_processor_cls:
                mock_processor = AsyncMock()
                mock_processor.bind = AsyncMock()
                mock_processor_cls.return_value = mock_processor

                with patch("aichat.pipeline.manager.Memory") as mock_memory_cls:
                    mock_memory = Mock()
                    mock_memory_cls.return_value = mock_memory

                    # Act
                    result = await manager.register(mock_chat, sdp_offer, mock_websocket, test_db)

                    # Assert - Memory created with correct args
                    mock_memory_cls.assert_called_once_with(
                        chat=mock_chat,
                        db=test_db,
                        ws=mock_websocket
                    )

                    # Assert - Processor created with correct args
                    mock_processor_cls.assert_called_once()
                    call_kwargs = mock_processor_cls.call_args.kwargs
                    assert call_kwargs["speech"] == "dummy"
                    assert call_kwargs["video"] == "dummy"
                    assert call_kwargs["llm"] == "dummy"
                    assert call_kwargs["voice"] == "test_voice"
                    assert call_kwargs["memory"] == mock_memory

                    # Assert - Processor bound to RTC and WebSocket
                    mock_processor.bind.assert_called_once_with(
                        rtc_in=mock_rtc_peer_connection,
                        ws_out=mock_websocket
                    )

                    # Assert - Connection registered
                    assert mock_chat.id in manager._conns
                    rtc, proc = manager._conns[mock_chat.id]
                    assert rtc == mock_rtc_peer_connection
                    assert proc == mock_processor

                    # Assert - SDP exchange completed
                    mock_rtc_peer_connection.setRemoteDescription.assert_called_once()
                    mock_rtc_peer_connection.createAnswer.assert_called_once()
                    mock_rtc_peer_connection.setLocalDescription.assert_called_once()
                    assert result == mock_rtc_peer_connection.localDescription

    @pytest.mark.asyncio
    async def test_register_sets_up_connection_state_change_handler(
        self, manager, mock_chat, mock_websocket, mock_rtc_peer_connection, test_db
    ):
        """Should register connectionstatechange handler to monitor RTC state."""
        with patch("aichat.pipeline.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection):
            with patch("aichat.pipeline.manager.Processor", return_value=AsyncMock()):
                with patch("aichat.pipeline.manager.Memory"):
                    await manager.register(mock_chat, "sdp", mock_websocket, test_db)

                    # Assert - Event handler registered
                    assert "connectionstatechange" in mock_rtc_peer_connection._event_handlers

    @pytest.mark.asyncio
    async def test_connection_state_failed_triggers_deregister(
        self, manager, mock_chat, mock_websocket, mock_rtc_peer_connection, test_db
    ):
        """Should automatically deregister when connection fails."""
        with patch("aichat.pipeline.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection):
            mock_processor = AsyncMock()
            with patch("aichat.pipeline.manager.Processor", return_value=mock_processor):
                with patch("aichat.pipeline.manager.Memory"):
                    await manager.register(mock_chat, "sdp", mock_websocket, test_db)

                    # Simulate state change to failed
                    mock_rtc_peer_connection.connectionState = "failed"
                    handler = mock_rtc_peer_connection._event_handlers["connectionstatechange"]
                    await handler()

                    # Assert - Connection cleaned up
                    assert mock_chat.id not in manager._conns
                    mock_rtc_peer_connection.close.assert_called_once()
                    mock_processor.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_state_closed_triggers_deregister(
        self, manager, mock_chat, mock_websocket, mock_rtc_peer_connection, test_db
    ):
        """Should automatically deregister when connection closes."""
        with patch("aichat.pipeline.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection):
            mock_processor = AsyncMock()
            with patch("aichat.pipeline.manager.Processor", return_value=mock_processor):
                with patch("aichat.pipeline.manager.Memory"):
                    await manager.register(mock_chat, "sdp", mock_websocket, test_db)

                    # Simulate state change to closed
                    mock_rtc_peer_connection.connectionState = "closed"
                    handler = mock_rtc_peer_connection._event_handlers["connectionstatechange"]
                    await handler()

                    # Assert - Connection cleaned up
                    assert mock_chat.id not in manager._conns
                    mock_rtc_peer_connection.close.assert_called_once()
                    mock_processor.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_state_disconnected_triggers_deregister(
        self, manager, mock_chat, mock_websocket, mock_rtc_peer_connection, test_db
    ):
        """Should automatically deregister when connection disconnects."""
        with patch("aichat.pipeline.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection):
            mock_processor = AsyncMock()
            with patch("aichat.pipeline.manager.Processor", return_value=mock_processor):
                with patch("aichat.pipeline.manager.Memory"):
                    await manager.register(mock_chat, "sdp", mock_websocket, test_db)

                    # Simulate state change to disconnected
                    mock_rtc_peer_connection.connectionState = "disconnected"
                    handler = mock_rtc_peer_connection._event_handlers["connectionstatechange"]
                    await handler()

                    # Assert - Connection cleaned up
                    assert mock_chat.id not in manager._conns
                    mock_rtc_peer_connection.close.assert_called_once()
                    mock_processor.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_state_connected_sends_avatar_initialization(
        self, manager, mock_chat, mock_websocket, mock_rtc_peer_connection, test_db
    ):
        """Should send avatar initialization message when connection established."""
        with patch("aichat.pipeline.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection):
            with patch("aichat.pipeline.manager.Processor", return_value=AsyncMock()):
                with patch("aichat.pipeline.manager.Memory"):
                    with patch("aichat.pipeline.manager.ModelFactory") as mock_factory:
                        mock_factory.get_voice.return_value = {"name": "test_voice", "path": "/voice.mp3"}
                        mock_factory.get_avatar.return_value = {"name": "test_face", "path": "/face.png"}

                        await manager.register(mock_chat, "sdp", mock_websocket, test_db)

                        # Simulate state change to connected
                        mock_rtc_peer_connection.connectionState = "connected"
                        handler = mock_rtc_peer_connection._event_handlers["connectionstatechange"]
                        await handler()

                        # Assert - Avatar initialization sent
                        mock_websocket.send_json.assert_called_with({
                            "type": MESSAGE_TYPE_AVATAR_INITIALIZE,
                            "data": {
                                "voice": {"name": "test_voice", "path": "/voice.mp3"},
                                "avatar": {"name": "test_face", "path": "/face.png"}
                            }
                        })

    @pytest.mark.asyncio
    async def test_deregister_closes_rtc_and_processor(
        self, manager, mock_chat, mock_websocket, mock_rtc_peer_connection, test_db
    ):
        """Should close both RTC and Processor on deregister."""
        with patch("aichat.pipeline.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection):
            mock_processor = AsyncMock()
            with patch("aichat.pipeline.manager.Processor", return_value=mock_processor):
                with patch("aichat.pipeline.manager.Memory"):
                    await manager.register(mock_chat, "sdp", mock_websocket, test_db)

                    # Act
                    await manager.deregister(mock_chat.id)

                    # Assert - Both closed
                    mock_rtc_peer_connection.close.assert_called_once()
                    mock_processor.close.assert_called_once()

                    # Assert - Connection removed
                    assert mock_chat.id not in manager._conns

    @pytest.mark.asyncio
    async def test_deregister_nonexistent_connection_raises_assertion(self, manager):
        """Should raise assertion error when deregistering unknown connection."""
        with pytest.raises(AssertionError, match="connection id not found"):
            await manager.deregister(999)

    @pytest.mark.asyncio
    async def test_multiple_connections_can_coexist(
        self, manager, mock_websocket, test_db
    ):
        """Should handle multiple concurrent connections."""
        chat1 = Mock(spec=Chat, id=1, voice="voice1", face="face1", llm="dummy", transcripts=[])
        chat2 = Mock(spec=Chat, id=2, voice="voice2", face="face2", llm="dummy", transcripts=[])

        mock_rtc1 = Mock()
        mock_rtc1.connectionState = "new"
        mock_rtc1.close = AsyncMock()
        mock_rtc1.setRemoteDescription = AsyncMock()
        mock_rtc1.createAnswer = AsyncMock()
        mock_rtc1.setLocalDescription = AsyncMock()
        mock_rtc1.localDescription = Mock(sdp="answer1")
        mock_rtc1._event_handlers = {}
        mock_rtc1.on = lambda event: lambda func: mock_rtc1._event_handlers.__setitem__(event, func) or func

        mock_rtc2 = Mock()
        mock_rtc2.connectionState = "new"
        mock_rtc2.close = AsyncMock()
        mock_rtc2.setRemoteDescription = AsyncMock()
        mock_rtc2.createAnswer = AsyncMock()
        mock_rtc2.setLocalDescription = AsyncMock()
        mock_rtc2.localDescription = Mock(sdp="answer2")
        mock_rtc2._event_handlers = {}
        mock_rtc2.on = lambda event: lambda func: mock_rtc2._event_handlers.__setitem__(event, func) or func

        with patch("aichat.pipeline.manager.RTCPeerConnection", side_effect=[mock_rtc1, mock_rtc2]):
            with patch("aichat.pipeline.manager.Processor", return_value=AsyncMock()):
                with patch("aichat.pipeline.manager.Memory"):
                    # Register both
                    await manager.register(chat1, "sdp1", mock_websocket, test_db)
                    await manager.register(chat2, "sdp2", mock_websocket, test_db)

                    # Assert - Both registered
                    assert 1 in manager._conns
                    assert 2 in manager._conns

                    # Deregister one
                    await manager.deregister(1)

                    # Assert - Only second remains
                    assert 1 not in manager._conns
                    assert 2 in manager._conns

    @pytest.mark.asyncio
    async def test_connection_state_change_ignores_invalid_states_when_not_registered(
        self, manager, mock_chat, mock_websocket, mock_rtc_peer_connection, test_db
    ):
        """Should not crash if state change fires after manual deregistration."""
        with patch("aichat.pipeline.manager.RTCPeerConnection", return_value=mock_rtc_peer_connection):
            mock_processor = AsyncMock()
            with patch("aichat.pipeline.manager.Processor", return_value=mock_processor):
                with patch("aichat.pipeline.manager.Memory"):
                    await manager.register(mock_chat, "sdp", mock_websocket, test_db)

                    # Manually deregister
                    await manager.deregister(mock_chat.id)

                    # Reset mock call counts
                    mock_rtc_peer_connection.close.reset_mock()
                    mock_processor.close.reset_mock()

                    # Simulate state change after deregistration
                    mock_rtc_peer_connection.connectionState = "failed"
                    handler = mock_rtc_peer_connection._event_handlers["connectionstatechange"]
                    await handler()

                    # Assert - No additional close calls (connection not in registry)
                    mock_rtc_peer_connection.close.assert_not_called()
                    mock_processor.close.assert_not_called()
