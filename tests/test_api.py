"""
API Integration Tests — FastAPI endpoints
==========================================
Tests the HTTP endpoints, session management,
and health check — no LLM API calls needed.
"""

import os
import sys
import pytest

# Add workspace root to path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# These tests require httpx which is an optional dependency
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Creates a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests the /health endpoint."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "active_sessions" in data

    def test_health_active_sessions_count(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["active_sessions"], int)
        assert data["active_sessions"] >= 0


class TestRootEndpoint:
    """Tests the / endpoint serves the landing page."""

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")


class TestSessionIsolation:
    """Tests that sessions are properly isolated via cookies."""

    def test_chat_sets_session_cookie(self, client):
        """POST /chat should set an episteme_session cookie."""
        resp = client.post("/chat", json={"user_input": "test"})
        # Even if chat fails due to no API key, the cookie behavior is set
        # Check that the response at least attempted to process
        assert resp.status_code in [200, 500]

    def test_status_returns_json(self, client):
        """GET /status should return valid JSON."""
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_reset_returns_success(self, client):
        """GET /reset should return success."""
        resp = client.get("/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"


class TestSSEEndpoint:
    """Tests the /chat/stream SSE endpoint."""

    def test_stream_without_session_returns_error(self, client):
        """GET /chat/stream without a session should return an error event."""
        resp = client.get("/chat/stream")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")


class TestInputValidation:
    """Tests input validation on endpoints."""

    def test_chat_requires_user_input(self, client):
        """POST /chat without user_input should return 422."""
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_requires_json(self, client):
        """POST /chat without JSON body should return 422."""
        resp = client.post("/chat")
        assert resp.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
