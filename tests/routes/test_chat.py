"""
Testing Strategy:
- Chat creation with authentication
- Listing user's chats
- Retrieving specific chat by ID
- Authorization checks (users can only access their own chats)
- Error handling for invalid requests
- Database interaction with in-memory SQLite
"""
from sqlmodel import select

from aichat.db_models.chat import Chat
from aichat.security.auth import SESSION_COOKIE_NAME


class TestCreateChat:
    """Test suite for POST /chat endpoint."""

    def test_create_chat_successfully(self, test_client, test_db):
        """Should create new chat and return chat data."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "chatuser",
                "password": "password123",
                "name": "Chat User",
                "bio": None
            }
        )

        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Create chat
        response = test_client.post(
            "/api/chat",
            json={
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "You are a helpful assistant"
                },
                "name": "test_name",
                "dialogue": "tiny_llama"
            },
            cookies=cookies
        )

        assert response.status_code == 200
        data = response.json()
        assert data["voice"] == "af_bella"
        assert data["face"] == "julia"
        assert data["prompt"] == "You are a helpful assistant"
        assert data["id"] is not None
        assert data["user_id"] == register_response.json()["id"]
        assert data["transcripts"] == []

        # Verify chat was created in database
        chat = test_db.exec(select(Chat).where(Chat.id == data["id"])).first()
        assert chat is not None
        assert chat.name == "test_name"
        assert chat.voice == "af_bella"
        assert chat.face == "julia"
        assert chat.prompt == "You are a helpful assistant"

    def test_create_chat_with_empty_prompt(self, test_client, test_db):
        """Should create chat with empty prompt."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "emptypromptuser",
                "password": "password123",
                "name": "Empty Prompt User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.post(
            "/api/chat",
            json={
                "name": "test_name",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": ""
                }
            },
            cookies=cookies
        )

        assert response.status_code == 200
        data = response.json()
        assert data["prompt"] == ""

    def test_create_chat_requires_authentication(self, test_client):
        """Should return 401 when not authenticated."""
        response = test_client.post(
            "/api/chat",
            json={
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Test prompt"
                }
            }
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_create_chat_with_invalid_session(self, test_client):
        """Should return 401 for invalid session token."""
        cookies = {SESSION_COOKIE_NAME: "invalid_token_here"}
        response = test_client.post(
            "/api/chat",
            json={
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Test prompt"
                }
            },
            cookies=cookies
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid session"

    def test_create_multiple_chats_for_same_user(self, test_client, test_db):
        """Should allow creating multiple chats for the same user."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "multichatsuser",
                "password": "password123",
                "name": "Multi Chats User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}
        user_id = register_response.json()["id"]

        # Create first chat
        response1 = test_client.post(
            "/api/chat",
            json={
                "name": "test_name",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "First chat"
                }
            },
            cookies=cookies
        )

        # Create second chat
        response2 = test_client.post(
            "/api/chat",
            json={
                "name": "test_name_2",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Second chat"
                }
            },
            cookies=cookies
        )

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["id"] != response2.json()["id"]

        # Verify both chats exist in database
        chats = test_db.exec(select(Chat).where(Chat.user_id == user_id)).all()
        assert len(chats) == 2


class TestGetChats:
    """Test suite for GET /chats endpoint."""

    def test_get_chats_returns_user_chats(self, test_client, test_db):
        """Should return all chats for authenticated user."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "getchatsuser",
                "password": "password123",
                "name": "Get Chats User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Create multiple chats
        test_client.post(
            "/api/chat",
            json={
                "name": "test1",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Chat 1"
                }
            },
            cookies=cookies
        )
        test_client.post(
            "/api/chat",
            json={
                "name": "test2",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Chat 2"
                }
            },
            cookies=cookies
        )

        # Get all chats
        response = test_client.get("/api/chats", cookies=cookies)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "test1"
        assert data[0]["prompt"] == "Chat 1"
        assert data[1]["name"] == "test2"
        assert data[1]["prompt"] == "Chat 2"

    def test_get_chats_returns_empty_list_for_new_user(self, test_client):
        """Should return empty list when user has no chats."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "nochatsuser",
                "password": "password123",
                "name": "No Chats User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.get("/api/chats", cookies=cookies)

        assert response.status_code == 200
        assert response.json() == []

    def test_get_chats_requires_authentication(self, test_client):
        """Should return 401 when not authenticated."""
        response = test_client.get("/api/chats")

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_get_chats_only_returns_own_chats(self, test_client, test_db):
        """Should only return chats belonging to authenticated user."""
        # Register first user
        register_response1 = test_client.post(
            "/api/register",
            json={
                "username": "user1",
                "password": "password123",
                "name": "User One",
                "bio": None
            }
        )
        cookies1 = {SESSION_COOKIE_NAME: register_response1.cookies[SESSION_COOKIE_NAME]}

        # Create chat for first user
        test_client.post(
            "/api/chat",
            json={
                "name": "test1",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "User 1 chat"
                }
            },
            cookies=cookies1
        )

        # Register second user
        register_response2 = test_client.post(
            "/api/register",
            json={
                "username": "user2",
                "password": "password123",
                "name": "User Two",
                "bio": None
            }
        )
        cookies2 = {SESSION_COOKIE_NAME: register_response2.cookies[SESSION_COOKIE_NAME]}

        # Create chat for second user
        test_client.post(
            "/api/chat",
            json={
                "name": "test2",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "User 2 chat"
                }
            },
            cookies=cookies2
        )

        # Get chats for second user
        response = test_client.get("/api/chats", cookies=cookies2)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["prompt"] == "User 2 chat"


class TestGetChatById:
    """Test suite for GET /chat/{id} endpoint."""

    def test_get_chat_by_id_successfully(self, test_client, test_db):
        """Should return specific chat by ID."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "getchatuser",
                "password": "password123",
                "name": "Get Chat User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Create chat
        create_response = test_client.post(
            "/api/chat",
            json={
                "name": "test",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Specific chat"
                }
            },
            cookies=cookies
        )
        chat_id = create_response.json()["id"]

        # Get chat by ID
        response = test_client.get(f"/api/chat/{chat_id}", cookies=cookies)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == chat_id
        assert data["prompt"] == "Specific chat"
        assert data["voice"] == "af_bella"
        assert data["face"] == "julia"

    def test_get_chat_by_id_not_found(self, test_client):
        """Should return 404 for non-existent chat."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "notfounduser",
                "password": "password123",
                "name": "Not Found User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Try to get non-existent chat
        response = test_client.get("/api/chat/99999", cookies=cookies)

        assert response.status_code == 404
        assert response.json()["detail"] == "Chat not found."

    def test_get_chat_by_id_requires_authentication(self, test_client):
        """Should return 401 when not authenticated."""
        response = test_client.get("/api/chat/1")

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_get_chat_by_id_requires_ownership(self, test_client, test_db):
        """Should return 404 when trying to access another user's chat."""
        # Register first user and create chat
        register_response1 = test_client.post(
            "/api/register",
            json={
                "username": "owner",
                "password": "password123",
                "name": "Owner User",
                "bio": None
            }
        )
        cookies1 = {SESSION_COOKIE_NAME: register_response1.cookies[SESSION_COOKIE_NAME]}

        create_response = test_client.post(
            "/api/chat",
            json={
                "name": "test",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Owner's chat"
                }
            },
            cookies=cookies1
        )
        chat_id = create_response.json()["id"]

        # Register second user
        register_response2 = test_client.post(
            "/api/register",
            json={
                "username": "intruder",
                "password": "password123",
                "name": "Intruder User",
                "bio": None
            }
        )
        cookies2 = {SESSION_COOKIE_NAME: register_response2.cookies[SESSION_COOKIE_NAME]}

        # Try to access first user's chat
        response = test_client.get(f"/api/chat/{chat_id}", cookies=cookies2)

        assert response.status_code == 404
        assert response.json()["detail"] == "Chat not found."


class TestRequestValidation:
    """Test suite for request validation."""

    def test_create_chat_with_missing_agent(self, test_client):
        """Should return 422 when agent is missing."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "validationuser",
                "password": "password123",
                "name": "Validation User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.post(
            "/api/chat",
            json={},
            cookies=cookies
        )

        assert response.status_code == 422

    def test_create_chat_with_missing_voice(self, test_client):
        """Should use default voice when voice is missing."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "defaultvoiceuser",
                "password": "password123",
                "name": "Default Voice User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.post(
            "/api/chat",
            json={
                "name": 'test',
                "agent": {
                    "face": "julia",
                    "prompt": "Test"
                }
            },
            cookies=cookies
        )

        # Should use default value from Agent schema
        assert response.status_code == 200
        assert response.json()["voice"] == "af_bella"

    def test_create_chat_with_missing_face(self, test_client):
        """Should use default face when face is missing."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "defaultfaceuser",
                "password": "password123",
                "name": "Default Face User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.post(
            "/api/chat",
            json={
                "name": "test",
                "agent": {
                    "voice": "af_bella",
                    "prompt": "Test"
                }
            },
            cookies=cookies
        )

        # Should use default value from Agent schema
        assert response.status_code == 200
        assert response.json()["face"] == "julia"

    def test_create_chat_with_missing_prompt(self, test_client):
        """Should use default empty prompt when prompt is missing."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "defaultpromptuser",
                "password": "password123",
                "name": "Default Prompt User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.post(
            "/api/chat",
            json={
                "name": "test",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia"
                }
            },
            cookies=cookies
        )

        # Should use default value from Agent schema
        assert response.status_code == 200
        assert response.json()["prompt"] == ""

    def test_get_chat_with_invalid_id(self, test_client):
        """Should return 422 for invalid chat ID format."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "invalidid",
                "password": "password123",
                "name": "Invalid ID User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.get("/api/chat/invalid", cookies=cookies)

        assert response.status_code == 422


class TestChatFlows:
    """Test suite for complete chat workflows."""

    def test_full_chat_creation_and_retrieval_flow(self, test_client, test_db):
        """Should complete full chat creation and retrieval flow."""
        # Register
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "flowuser",
                "password": "password123",
                "name": "Flow User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Create chat
        create_response = test_client.post(
            "/api/chat",
            json={
                "name": "test",
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": "Flow test chat"
                }
            },
            cookies=cookies
        )
        assert create_response.status_code == 200
        chat_id = create_response.json()["id"]

        # Get all chats
        list_response = test_client.get("/api/chats", cookies=cookies)
        assert list_response.status_code == 200
        assert len(list_response.json()) == 1

        # Get specific chat
        get_response = test_client.get(f"/api/chat/{chat_id}", cookies=cookies)
        assert get_response.status_code == 200
        assert get_response.json()["id"] == chat_id
        assert get_response.json()["prompt"] == "Flow test chat"

    def test_create_multiple_chats_and_list_them(self, test_client, test_db):
        """Should create multiple chats and list them all."""
        # Register
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "multichatuser",
                "password": "password123",
                "name": "Multi Chat User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Create 3 chats
        chat_ids = []
        for i in range(3):
            response = test_client.post(
                "/api/chat",
                json={
                    "name": "test",
                    "agent": {
                        "voice": "af_bella",
                        "face": "julia",
                        "prompt": f"Chat {i + 1}"
                    }
                },
                cookies=cookies
            )
            assert response.status_code == 200
            chat_ids.append(response.json()["id"])

        # List all chats
        list_response = test_client.get("/api/chats", cookies=cookies)
        assert list_response.status_code == 200
        chats = list_response.json()
        assert len(chats) == 3

        # Verify all chat IDs are present
        returned_ids = [chat["id"] for chat in chats]
        assert set(returned_ids) == set(chat_ids)


class TestEdgeCases:
    """Test suite for edge cases and error scenarios."""

    def test_create_chat_with_very_long_prompt(self, test_client, test_db):
        """Should handle very long prompt text."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "longpromptuser",
                "password": "password123",
                "name": "Long Prompt User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        long_prompt = "A" * 10000
        response = test_client.post(
            "/api/chat",
            json={
                "agent": {
                    "voice": "af_bella",
                    "face": "julia",
                    "prompt": long_prompt
                }
            },
            cookies=cookies
        )

        # Should work or fail gracefully
        assert response.status_code in [200, 413, 422]
        if response.status_code == 200:
            assert response.json()["prompt"] == long_prompt

    def test_get_chat_with_zero_id(self, test_client):
        """Should handle chat ID of 0."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "zeroiduser",
                "password": "password123",
                "name": "Zero ID User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.get("/api/chat/0", cookies=cookies)

        # Should return 404 as ID 0 typically doesn't exist
        assert response.status_code == 404

    def test_get_chat_with_negative_id(self, test_client):
        """Should handle negative chat ID."""
        # Register and get session
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "negativeiduser",
                "password": "password123",
                "name": "Negative ID User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        response = test_client.get("/api/chat/-1", cookies=cookies)

        # Should return 404 or 422
        assert response.status_code in [404, 422]

    def test_session_persists_across_chat_operations(self, test_client, test_db):
        """Should maintain session across multiple chat operations."""
        # Register
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "persistchatuser",
                "password": "password123",
                "name": "Persist Chat User",
                "bio": None
            }
        )
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Multiple operations with same session
        create1 = test_client.post(
            "/api/chat",
            json={"agent": {"voice": "af_bella", "face": "julia", "prompt": "Chat 1"}, "name": "test1"},
            cookies=cookies
        )
        create2 = test_client.post(
            "/api/chat",
            json={"agent": {"voice": "af_bella", "face": "julia", "prompt": "Chat 2"}, "name": "test2"},
            cookies=cookies
        )
        list_chats = test_client.get("/api/chats", cookies=cookies)
        get_chat = test_client.get(f"/api/chat/{create1.json()['id']}", cookies=cookies)

        assert create1.status_code == 200
        assert create2.status_code == 200
        assert list_chats.status_code == 200
        assert get_chat.status_code == 200
        assert len(list_chats.json()) == 2
