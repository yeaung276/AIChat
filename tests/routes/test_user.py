"""
Testing Strategy:
- User registration flow with validation
- User login authentication and token generation
- Protected endpoint access with authentication
- Error handling for invalid credentials
- Database interaction with in-memory SQLite
"""
import pytest
from sqlmodel import select

from aichat.db_models.user import User
from aichat.security.auth import SESSION_COOKIE_NAME


class TestUserRegistration:
    """Test suite for /register endpoint."""

    def test_register_creates_user_successfully(self, test_client, test_db):
        """Should create new user and return user data with session cookie."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "testuser",
                "password": "password123",
                "name": "Test User",
                "bio": "Test bio"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["screen_name"] == "Test User"
        assert data["bio"] == "Test bio"
        assert data["id"] is not None
        assert "pwd_hash" in data

        # Verify session cookie was set
        assert SESSION_COOKIE_NAME in response.cookies

        # Verify user was created in database
        user = test_db.exec(select(User).where(User.username == "testuser")).first()
        assert user is not None
        assert user.username == "testuser"
        assert user.screen_name == "Test User"

    def test_register_with_optional_bio(self, test_client, test_db):
        """Should create user without bio field."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "nobiouser",
                "password": "password123",
                "name": "No Bio User",
                "bio": None
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "nobiouser"
        assert data["bio"] is None

        # Verify in database
        user = test_db.exec(select(User).where(User.username == "nobiouser")).first()
        assert user is not None
        assert user.bio is None

    def test_register_hashes_password(self, test_client, test_db):
        """Should hash password before storing."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "secureuser",
                "password": "plaintext_password",
                "name": "Secure User",
                "bio": None
            }
        )

        assert response.status_code == 200

        # Verify password is hashed in database
        user = test_db.exec(select(User).where(User.username == "secureuser")).first()
        assert user is not None
        assert user.pwd_hash != "plaintext_password"
        assert user.pwd_hash.startswith("$2b$")  # bcrypt hash prefix

    def test_register_sets_session_cookie(self, test_client):
        """Should set session cookie with proper attributes."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "cookieuser",
                "password": "password123",
                "name": "Cookie User",
                "bio": None
            }
        )

        assert response.status_code == 200
        assert SESSION_COOKIE_NAME in response.cookies
        cookie_value = response.cookies[SESSION_COOKIE_NAME]
        assert len(cookie_value) > 0

    def test_register_with_duplicate_username(self, test_client, test_db):
        """Should handle duplicate username gracefully."""
        # Create first user
        test_client.post(
            "/api/register",
            json={
                "username": "duplicate",
                "password": "password123",
                "name": "First User",
                "bio": None
            }
        )

        # Try to create duplicate - should fail with database integrity error
        response = test_client.post(
            "/api/register",
            json={
                "username": "duplicate",
                "password": "different_password",
                "name": "Second User",
                "bio": None
            }
        )

        # Should fail with server error due to unique constraint violation
        assert response.status_code == 400


class TestUserLogin:
    """Test suite for /login endpoint."""

    def test_login_with_valid_credentials(self, test_client, test_db):
        """Should authenticate user and return user data with session cookie."""
        # First register a user
        test_client.post(
            "/api/register",
            json={
                "username": "loginuser",
                "password": "password123",
                "name": "Login User",
                "bio": "Test bio"
            }
        )

        # Now login
        response = test_client.post(
            "/api/login",
            json={
                "username": "loginuser",
                "password": "password123"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "loginuser"
        assert data["screen_name"] == "Login User"

        # Verify session cookie was set
        assert SESSION_COOKIE_NAME in response.cookies

    def test_login_with_invalid_username(self, test_client):
        """Should return 401 for non-existent user."""
        response = test_client.post(
            "/api/login",
            json={
                "username": "nonexistent",
                "password": "password123"
            }
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_with_invalid_password(self, test_client, test_db):
        """Should return 401 for incorrect password."""
        # Register user
        test_client.post(
            "/api/register",
            json={
                "username": "wrongpwd",
                "password": "correct_password",
                "name": "Wrong Password User",
                "bio": None
            }
        )

        # Try to login with wrong password
        response = test_client.post(
            "/api/login",
            json={
                "username": "wrongpwd",
                "password": "wrong_password"
            }
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_login_creates_valid_session_token(self, test_client, test_db):
        """Should create session token that can be used for authentication."""
        # Register and login
        test_client.post(
            "/api/register",
            json={
                "username": "tokenuser",
                "password": "password123",
                "name": "Token User",
                "bio": None
            }
        )

        login_response = test_client.post(
            "/api/login",
            json={
                "username": "tokenuser",
                "password": "password123"
            }
        )

        assert login_response.status_code == 200
        token = login_response.cookies.get(SESSION_COOKIE_NAME)
        assert token is not None
        assert len(token) > 0


class TestGetCurrentUser:
    """Test suite for /me endpoint."""

    def test_me_returns_authenticated_user(self, test_client, test_db):
        """Should return current user data when authenticated."""
        # Register user
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "meuser",
                "password": "password123",
                "name": "Me User",
                "bio": "My bio"
            }
        )

        # Get session cookie
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Access /me endpoint
        response = test_client.get("/api/me", cookies=cookies)

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "meuser"
        assert data["screen_name"] == "Me User"
        assert data["bio"] == "My bio"

    def test_me_requires_authentication(self, test_client):
        """Should return 401 when not authenticated."""
        response = test_client.get("/api/me")

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_me_with_invalid_session(self, test_client):
        """Should return 401 for invalid session token."""
        cookies = {SESSION_COOKIE_NAME: "invalid_token_here"}
        response = test_client.get("/api/me", cookies=cookies)

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid session"

    def test_me_with_expired_token(self, test_client, test_db):
        """Should return 401 for expired session token."""
        # Create an expired token
        from aichat.security.auth import create_session_token
        from datetime import datetime, timedelta
        from jose import jwt
        import os

        # Register a user first
        test_client.post(
            "/api/register",
            json={
                "username": "expireduser",
                "password": "password123",
                "name": "Expired User",
                "bio": None
            }
        )

        # Create expired token
        user = test_db.exec(select(User).where(User.username == "expireduser")).first()
        expired_payload = {
            "sub": str(user.id),
            "exp": datetime.utcnow() - timedelta(minutes=10),  # Expired 10 minutes ago
            "type": "session",
        }
        expired_token = jwt.encode(
            expired_payload,
            os.getenv("JWT_SECRET", "secret-key"),
            algorithm="HS256"
        )

        cookies = {SESSION_COOKIE_NAME: expired_token}
        response = test_client.get("/api/me", cookies=cookies)

        assert response.status_code == 401


class TestAuthenticationFlow:
    """Test suite for complete authentication flows."""

    def test_full_registration_flow(self, test_client, test_db):
        """Should complete full registration and authentication flow."""
        # Register
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "fullflowuser",
                "password": "password123",
                "name": "Full Flow User",
                "bio": "Testing full flow"
            }
        )

        assert register_response.status_code == 200
        assert SESSION_COOKIE_NAME in register_response.cookies

        # Access protected endpoint with session
        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}
        me_response = test_client.get("/api/me", cookies=cookies)

        assert me_response.status_code == 200
        assert me_response.json()["username"] == "fullflowuser"

    def test_full_login_flow(self, test_client, test_db):
        """Should complete full login and authentication flow."""
        # Register user first
        test_client.post(
            "/api/register",
            json={
                "username": "loginflowuser",
                "password": "password123",
                "name": "Login Flow User",
                "bio": "Testing login flow"
            }
        )

        # Login
        login_response = test_client.post(
            "/api/login",
            json={
                "username": "loginflowuser",
                "password": "password123"
            }
        )

        assert login_response.status_code == 200
        assert SESSION_COOKIE_NAME in login_response.cookies

        # Access protected endpoint
        cookies = {SESSION_COOKIE_NAME: login_response.cookies[SESSION_COOKIE_NAME]}
        me_response = test_client.get("/api/me", cookies=cookies)

        assert me_response.status_code == 200
        assert me_response.json()["username"] == "loginflowuser"

    def test_session_persists_across_requests(self, test_client, test_db):
        """Should maintain session across multiple requests."""
        # Register
        register_response = test_client.post(
            "/api/register",
            json={
                "username": "persistuser",
                "password": "password123",
                "name": "Persist User",
                "bio": None
            }
        )

        cookies = {SESSION_COOKIE_NAME: register_response.cookies[SESSION_COOKIE_NAME]}

        # Make multiple requests with same session
        response1 = test_client.get("/api/me", cookies=cookies)
        response2 = test_client.get("/api/me", cookies=cookies)
        response3 = test_client.get("/api/me", cookies=cookies)

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response3.status_code == 200
        assert response1.json()["username"] == "persistuser"
        assert response2.json()["username"] == "persistuser"
        assert response3.json()["username"] == "persistuser"


class TestRequestValidation:
    """Test suite for request validation."""

    def test_register_with_missing_username(self, test_client):
        """Should return 422 when username is missing."""
        response = test_client.post(
            "/api/register",
            json={
                "password": "password123",
                "name": "Test User",
                "bio": None
            }
        )

        assert response.status_code == 422

    def test_register_with_missing_password(self, test_client):
        """Should return 422 when password is missing."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "testuser",
                "name": "Test User",
                "bio": None
            }
        )

        assert response.status_code == 422

    def test_register_with_missing_name(self, test_client):
        """Should return 422 when name is missing."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "testuser",
                "password": "password123",
                "bio": None
            }
        )

        assert response.status_code == 422

    def test_login_with_missing_username(self, test_client):
        """Should return 422 when username is missing."""
        response = test_client.post(
            "/api/login",
            json={"password": "password123"}
        )

        assert response.status_code == 422

    def test_login_with_missing_password(self, test_client):
        """Should return 422 when password is missing."""
        response = test_client.post(
            "/api/login",
            json={"username": "testuser"}
        )

        assert response.status_code == 422

    def test_login_with_empty_body(self, test_client):
        """Should return 422 for empty request body."""
        response = test_client.post("/api/login", json={})

        assert response.status_code == 422

    def test_register_with_empty_body(self, test_client):
        """Should return 422 for empty request body."""
        response = test_client.post("/api/register", json={})

        assert response.status_code == 422


class TestEdgeCases:
    """Test suite for edge cases and error scenarios."""

    def test_register_with_empty_username(self, test_client):
        """Should handle empty string username."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "",
                "password": "password123",
                "name": "Empty Username",
                "bio": None
            }
        )

        # Should either fail validation or database constraint
        assert response.status_code in [400, 422, 500]

    def test_register_with_empty_password(self, test_client):
        """Should handle empty string password."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "emptypwduser",
                "password": "",
                "name": "Empty Password",
                "bio": None
            }
        )

        # Could accept it (weak password) or reject it
        # Just verify it doesn't crash
        assert response.status_code in [200, 400, 422]

    def test_login_after_multiple_failed_attempts(self, test_client, test_db):
        """Should allow login after multiple failed attempts."""
        # Register user
        test_client.post(
            "/api/register",
            json={
                "username": "retryuser",
                "password": "correct_password",
                "name": "Retry User",
                "bio": None
            }
        )

        # Failed attempts
        for _ in range(3):
            response = test_client.post(
                "/api/login",
                json={
                    "username": "retryuser",
                    "password": "wrong_password"
                }
            )
            assert response.status_code == 401

        # Successful attempt
        response = test_client.post(
            "/api/login",
            json={
                "username": "retryuser",
                "password": "correct_password"
            }
        )
        assert response.status_code == 200

    def test_register_with_special_characters_in_username(self, test_client, test_db):
        """Should handle special characters in username."""
        response = test_client.post(
            "/api/register",
            json={
                "username": "user@test.com",
                "password": "password123",
                "name": "Special User",
                "bio": None
            }
        )

        # Should work or fail gracefully
        assert response.status_code in [200, 400, 422]

    def test_register_with_long_bio(self, test_client, test_db):
        """Should handle very long bio text."""
        long_bio = "A" * 10000
        response = test_client.post(
            "/api/register",
            json={
                "username": "longbiouser",
                "password": "password123",
                "name": "Long Bio User",
                "bio": long_bio
            }
        )

        # Should work or fail gracefully
        assert response.status_code in [200, 400, 413, 422]
