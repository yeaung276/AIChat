"""
Testing Strategy:
- WebSocket connection with valid SDP offer and chat ownership
- Chat validation (missing chat_id, non-existent chat, unauthorized access)
- ConnectionManager integration (register method called correctly)
- Error handling for invalid message types
"""
import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from sqlmodel import select

from aichat.types import MESSAGE_TYPE_SDP_ANSWER, MESSAGE_TYPE_SDP_OFFER
from aichat.db_models.chat import Chat
from aichat.security.auth import SESSION_COOKIE_NAME


class TestWebSocketSdpExchange:
    """Test suite for /ws WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_successful_sdp_exchange(self, test_client, test_db, mock_websocket):
        """Should accept SDP offer, register connection, and return SDP answer."""
        # Create user and chat
        register_response = test_client.post(
            "/register",
            json={
                "username": "wsuser",
                "password": "password123",
                "name": "WS User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        chat_response = test_client.post(
            "/chat",
            json={
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Test chat"
                },
                "dialogue": "tiny_llama"
            },
            cookies=cookies
        )
        chat_id = chat_response.json()["id"]

        # Mock ConnectionManager.register
        with patch("aichat.routes.chat.conn_mg.register") as mock_register:
            mock_answer = Mock()
            mock_answer.sdp = "mock_answer_sdp"
            mock_register.return_value = mock_answer

            # Setup websocket to receive one SDP offer
            offer_message = json.dumps({
                "type": MESSAGE_TYPE_SDP_OFFER,
                "chat_id": chat_id,
                "sdp": "test_offer_sdp"
            })

            async def mock_iter():
                yield offer_message

            mock_websocket.iter_text = Mock(return_value=mock_iter())

            # Import and call the handler
            from aichat.routes.chat import sdp_exchange

            # Mock the user and db dependencies
            user = register_response.json()
            chat = test_db.exec(select(Chat).where(Chat.id == chat_id)).first()

            await sdp_exchange(mock_websocket, user=Mock(id=user["id"]), db=test_db)

            # Verify connection was accepted
            mock_websocket.accept.assert_called_once()

            # Verify register was called with correct chat and SDP
            assert mock_register.call_count == 1
            call_args = mock_register.call_args
            assert call_args[0][0].id == chat_id
            assert call_args[0][1] == mock_websocket
            assert call_args[0][2] == "test_offer_sdp"

            # Verify answer was sent back
            mock_websocket.send_json.assert_called_once_with({
                "type": MESSAGE_TYPE_SDP_ANSWER,
                "sdp": "mock_answer_sdp"
            })

    @pytest.mark.asyncio
    async def test_missing_chat_id(self, test_client, test_db, mock_websocket):
        """Should return error when chat_id is missing."""
        # Create user
        register_response = test_client.post(
            "/register",
            json={
                "username": "nochatid",
                "password": "password123",
                "name": "No Chat ID User",
                "bio": None
            }
        )

        # Setup websocket to receive offer without chat_id
        offer_message = json.dumps({
            "type": MESSAGE_TYPE_SDP_OFFER,
            "sdp": "test_offer_sdp"
        })

        async def mock_iter():
            yield offer_message

        mock_websocket.iter_text = Mock(return_value=mock_iter())

        from aichat.routes.chat import sdp_exchange

        user = register_response.json()
        await sdp_exchange(mock_websocket, user=Mock(id=user["id"]), db=test_db)

        # Verify error was sent
        mock_websocket.send_json.assert_called_with({
            "type": "error",
            "message": "chat_id is required."
        })

    @pytest.mark.asyncio
    async def test_chat_not_found(self, test_client, test_db, mock_websocket):
        """Should return error when chat does not exist."""
        # Create user
        register_response = test_client.post(
            "/register",
            json={
                "username": "chatnotfound",
                "password": "password123",
                "name": "Chat Not Found User",
                "bio": None
            }
        )

        # Setup websocket with non-existent chat_id
        offer_message = json.dumps({
            "type": MESSAGE_TYPE_SDP_OFFER,
            "chat_id": 99999,
            "sdp": "test_offer_sdp"
        })

        async def mock_iter():
            yield offer_message

        mock_websocket.iter_text = Mock(return_value=mock_iter())

        from aichat.routes.chat import sdp_exchange

        user = register_response.json()
        await sdp_exchange(mock_websocket, user=Mock(id=user["id"]), db=test_db)

        # Verify error was sent
        mock_websocket.send_json.assert_called_with({
            "type": "error",
            "message": "chat not found."
        })

    @pytest.mark.asyncio
    async def test_unauthorized_chat_access(self, test_client, test_db, mock_websocket):
        """Should return error when user tries to access another user's chat."""
        # Create first user and their chat
        user1_response = test_client.post(
            "/register",
            json={
                "username": "owner",
                "password": "password123",
                "name": "Owner User",
                "bio": None
            }
        )
        cookies1 = {SESSION_COOKIE_NAME: user1_response.cookies[SESSION_COOKIE_NAME]}

        chat_response = test_client.post(
            "/chat",
            json={
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Owner's chat"
                },
                "dialogue": "tiny_llama"
            },
            cookies=cookies1
        )
        chat_id = chat_response.json()["id"]

        # Create second user (intruder)
        user2_response = test_client.post(
            "/register",
            json={
                "username": "intruder",
                "password": "password123",
                "name": "Intruder User",
                "bio": None
            }
        )

        # Setup websocket trying to access first user's chat
        offer_message = json.dumps({
            "type": MESSAGE_TYPE_SDP_OFFER,
            "chat_id": chat_id,
            "sdp": "test_offer_sdp"
        })

        async def mock_iter():
            yield offer_message

        mock_websocket.iter_text = Mock(return_value=mock_iter())

        from aichat.routes.chat import sdp_exchange

        user2 = user2_response.json()
        await sdp_exchange(mock_websocket, user=Mock(id=user2["id"]), db=test_db)

        # Verify error was sent (chat not found because of ownership check)
        mock_websocket.send_json.assert_called_with({
            "type": "error",
            "message": "chat not found."
        })

    @pytest.mark.asyncio
    async def test_unrecognized_message_type(self, test_client, test_db, mock_websocket):
        """Should return error for unrecognized message types."""
        # Create user
        register_response = test_client.post(
            "/register",
            json={
                "username": "unrecognized",
                "password": "password123",
                "name": "Unrecognized User",
                "bio": None
            }
        )

        # Setup websocket with invalid message type
        invalid_message = json.dumps({
            "type": "INVALID_TYPE",
            "data": "something"
        })

        async def mock_iter():
            yield invalid_message

        mock_websocket.iter_text = Mock(return_value=mock_iter())

        from aichat.routes.chat import sdp_exchange

        user = register_response.json()
        await sdp_exchange(mock_websocket, user=Mock(id=user["id"]), db=test_db)

        # Verify error was sent
        mock_websocket.send_json.assert_called_with({
            "type": "error",
            "message": "Unrecognized message type."
        })

    @pytest.mark.asyncio
    async def test_connection_manager_exception_handling(self, test_client, test_db, mock_websocket):
        """Should handle exceptions from ConnectionManager gracefully."""
        # Create user and chat
        register_response = test_client.post(
            "/register",
            json={
                "username": "exception",
                "password": "password123",
                "name": "Exception User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        chat_response = test_client.post(
            "/chat",
            json={
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Test chat"
                },
                "dialogue": "tiny_llama"
            },
            cookies=cookies
        )
        chat_id = chat_response.json()["id"]

        # Mock ConnectionManager.register to raise exception
        with patch("aichat.routes.chat.conn_mg.register") as mock_register:
            mock_register.side_effect = Exception("Connection failed")

            # Setup websocket to receive one SDP offer
            offer_message = json.dumps({
                "type": MESSAGE_TYPE_SDP_OFFER,
                "chat_id": chat_id,
                "sdp": "test_offer_sdp"
            })

            async def mock_iter():
                yield offer_message

            mock_websocket.iter_text = Mock(return_value=mock_iter())

            from aichat.routes.chat import sdp_exchange

            user = register_response.json()
            # Should not raise exception, it's caught and logged
            await sdp_exchange(mock_websocket, user=Mock(id=user["id"]), db=test_db)

            # Connection should still have been accepted
            mock_websocket.accept.assert_called_once()
