"""
Episteme API Server
====================
Production-grade FastAPI backend with:
- Session isolation via cookies
- SSE streaming for KAIROS analysis
- SQLite persistence
- Structured logging
"""

import os
import sys
import json
import time
import uuid
import secrets
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Response, Cookie
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

# Ensure workspace root and qila directory are in sys.path
workspace_root = os.path.dirname(os.path.abspath(__file__))
qila_dir = os.path.join(workspace_root, "qila")
if qila_dir not in sys.path:
    sys.path.insert(0, qila_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Load .env before any KAIROS imports
try:
    from dotenv import load_dotenv
    load_dotenv(Path(workspace_root) / ".env")
except ImportError:
    pass

from qila.qila_session import QILASession
from qila.logger import get_logger

log = get_logger("episteme.api")

app = FastAPI(
    title="Episteme API",
    description="Unified AI Epistemic Workspace — KAIROS + MAKS",
    version="1.0.0"
)

# ═══════════════════════════════════════════════
#  Session Management
# ═══════════════════════════════════════════════

# App secret for cookie signing (from env or generated)
APP_SECRET = os.environ.get("EPISTEME_SECRET", secrets.token_hex(32))

# Session store: maps session_id -> QILASession
_sessions: dict[str, QILASession] = {}
_session_last_active: dict[str, float] = {}
SESSION_TTL = 3600  # 1 hour inactivity timeout


def _get_or_create_session(session_id: Optional[str] = None) -> tuple[str, QILASession]:
    """Returns (session_id, session). Creates new if needed."""
    # Cleanup expired sessions
    now = time.time()
    expired = [sid for sid, t in _session_last_active.items() if now - t > SESSION_TTL]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_last_active.pop(sid, None)
        log.info(f"Expired session: {sid[:8]}...")

    if session_id and session_id in _sessions:
        _session_last_active[session_id] = now
        return session_id, _sessions[session_id]

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    session = QILASession(run_kairos=True, session_id=new_id)
    _sessions[new_id] = session
    _session_last_active[new_id] = now
    log.info(f"Created new session: {new_id[:8]}...")
    return new_id, session


def _session_from_request(request: Request) -> tuple[str, QILASession]:
    """Extracts session from cookie or creates a new one."""
    session_id = request.cookies.get("episteme_session")
    return _get_or_create_session(session_id)


class ChatRequest(BaseModel):
    user_input: str


# ═══════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════

@app.get("/")
def read_root():
    """Serves the landing page."""
    landing_path = os.path.join(workspace_root, "landing.html")
    if not os.path.exists(landing_path):
        raise HTTPException(status_code=404, detail="landing.html not found")
    return FileResponse(landing_path)


@app.get("/app")
def read_app(request: Request):
    """Serves the main workspace and provisions a session."""
    index_path = os.path.join(workspace_root, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    
    response = FileResponse(index_path)
    
    # Pre-provision the session cookie on page load
    # This prevents background polls from creating orphan sessions
    session_id, _ = _session_from_request(request)
    response.set_cookie(
        key="episteme_session",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL
    )
    return response


@app.post("/chat")
def chat(request_body: ChatRequest, request: Request, response: Response):
    """
    Phase 1 — Instant answer (~3-5 seconds).
    Returns the LLM response + memory state. No KAIROS analysis yet.
    Sets a session cookie for session isolation.
    """
    try:
        session_id, session = _session_from_request(request)
        result = session.chat_fast(request_body.user_input)

        # Set session cookie
        response.set_cookie(
            key="episteme_session",
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=SESSION_TTL
        )

        return result
    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stream")
def chat_stream(request: Request):
    """
    Phase 2 — SSE streaming endpoint for KAIROS claim analysis.
    Streams scored claims one at a time via Server-Sent Events.
    """
    session_id = request.cookies.get("episteme_session")
    if not session_id or session_id not in _sessions:
        def error_gen():
            yield 'event: error\ndata: {"type":"error","message":"No active session"}\n\n'
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    session = _sessions[session_id]

    def event_generator():
        try:
            for event in session.stream_analysis():
                event_type = event.get("type", "unknown")
                data = json.dumps(event)
                yield f"event: {event_type}\ndata: {data}\n\n"
        except Exception as e:
            log.error(f"Stream error: {e}")
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/status")
def status(request: Request):
    """Returns the live status metrics of the current session."""
    try:
        session_id = request.cookies.get("episteme_session")
        if not session_id or session_id not in _sessions:
            return {"active_memories": 0, "ghost_memories": 0, "db_size_kb": 0.0}
        
        session = _sessions[session_id]
        return session.status()
    except Exception as e:
        log.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reset")
def reset(request: Request):
    """Resets the session's memory store and counters."""
    try:
        session_id = request.cookies.get("episteme_session")
        if not session_id or session_id not in _sessions:
            return {"status": "ignored", "message": "No active session to reset"}
            
        session = _sessions[session_id]
        session.reset()
        return {"status": "success", "message": f"Session reset. ID: {session_id[:8]}..."}
    except Exception as e:
        log.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session_map")
def session_map(request: Request):
    """Returns the full conversation history for the session map view."""
    try:
        session_id = request.cookies.get("episteme_session")
        if not session_id or session_id not in _sessions:
            return []
            
        session = _sessions[session_id]
        return session.conversation_history
    except Exception as e:
        log.error(f"Session map error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint for Docker/deployment."""
    return {
        "status": "healthy",
        "active_sessions": len(_sessions),
        "uptime": "ok"
    }


if __name__ == "__main__":
    import uvicorn
    log.info("Starting Episteme server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
