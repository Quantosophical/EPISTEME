"""
Episteme Persistence Layer — SQLite Storage
============================================
Persists memory units, ghost store, and conversation history
to SQLite so data survives server restarts.
"""

import os
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Database location — project root
_DB_PATH = Path(__file__).resolve().parent.parent / "episteme.db"


def _get_connection(db_path: str = None) -> sqlite3.Connection:
    """Returns a SQLite connection with WAL mode for concurrent reads."""
    path = db_path or str(_DB_PATH)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = None):
    """Creates tables if they don't exist."""
    conn = _get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory_units (
            id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            original_content TEXT,
            fidelity TEXT NOT NULL DEFAULT 'FULL',
            created_at REAL NOT NULL,
            last_accessed_at REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            cached_survival REAL,
            cached_at REAL,
            entropy_score REAL DEFAULT 0.5,
            access_history TEXT DEFAULT '[]',
            connections TEXT DEFAULT '[]',
            PRIMARY KEY (id, session_id)
        );

        CREATE TABLE IF NOT EXISTS ghost_units (
            id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            original_content TEXT,
            fidelity TEXT NOT NULL DEFAULT 'GHOST',
            created_at REAL NOT NULL,
            last_accessed_at REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            cached_survival REAL,
            cached_at REAL,
            entropy_score REAL DEFAULT 0.5,
            access_history TEXT DEFAULT '[]',
            connections TEXT DEFAULT '[]',
            PRIMARY KEY (id, session_id)
        );

        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn INTEGER NOT NULL,
            user_input TEXT NOT NULL,
            answer TEXT NOT NULL,
            kairos_field TEXT DEFAULT '[]',
            uncertain_claims TEXT DEFAULT '[]',
            reconsolidated TEXT DEFAULT '[]',
            memory_count INTEGER DEFAULT 0,
            ghost_count INTEGER DEFAULT 0,
            pressure TEXT DEFAULT 'LOW',
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_units(session_id);
        CREATE INDEX IF NOT EXISTS idx_ghost_session ON ghost_units(session_id);
        CREATE INDEX IF NOT EXISTS idx_history_session ON conversation_history(session_id);
    """)
    conn.commit()
    conn.close()


def _unit_to_row(unit, session_id: str) -> dict:
    """Converts a MemoryUnit to a dict for SQLite insertion."""
    return {
        "id": unit.id,
        "session_id": session_id,
        "content": unit.content,
        "original_content": getattr(unit, "original_content", None),
        "fidelity": unit.fidelity,
        "created_at": unit.created_at,
        "last_accessed_at": unit.last_accessed_at,
        "access_count": unit.access_count,
        "cached_survival": unit.cached_survival,
        "cached_at": unit.cached_at,
        "entropy_score": unit.entropy_score,
        "access_history": json.dumps(unit.access_history),
        "connections": json.dumps(unit.connections),
    }


def _row_to_unit(row):
    """Converts a SQLite row back to a MemoryUnit."""
    # Import here to avoid circular imports
    from memory_unit import MemoryUnit
    unit = MemoryUnit(
        id=row["id"],
        content=row["content"],
        fidelity=row["fidelity"],
        created_at=row["created_at"],
        last_accessed_at=row["last_accessed_at"],
        access_count=row["access_count"],
        entropy_score=row["entropy_score"],
        cached_survival=row["cached_survival"],
        cached_at=row["cached_at"],
        access_history=json.loads(row["access_history"] or "[]"),
        connections=json.loads(row["connections"] or "[]"),
    )
    unit.original_content = row["original_content"]
    return unit


# ═══════════════════════════════════════════════
#  Memory Store Persistence
# ═══════════════════════════════════════════════

def save_memories(memory_store: dict, session_id: str, db_path: str = None):
    """Persists all memory units for a session (full replace)."""
    conn = _get_connection(db_path)
    conn.execute("DELETE FROM memory_units WHERE session_id = ?", (session_id,))
    for unit in memory_store.values():
        row = _unit_to_row(unit, session_id)
        conn.execute("""
            INSERT INTO memory_units 
            (id, session_id, content, original_content, fidelity, created_at,
             last_accessed_at, access_count, cached_survival, cached_at,
             entropy_score, access_history, connections)
            VALUES (:id, :session_id, :content, :original_content, :fidelity,
                    :created_at, :last_accessed_at, :access_count, :cached_survival,
                    :cached_at, :entropy_score, :access_history, :connections)
        """, row)
    conn.commit()
    conn.close()


def load_memories(session_id: str, db_path: str = None) -> dict:
    """Loads all memory units for a session. Returns {id: MemoryUnit}."""
    conn = _get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM memory_units WHERE session_id = ?", (session_id,)
    ).fetchall()
    conn.close()
    return {row["id"]: _row_to_unit(row) for row in rows}


# ═══════════════════════════════════════════════
#  Ghost Store Persistence
# ═══════════════════════════════════════════════

def save_ghosts(ghost_store_dict: dict, session_id: str, db_path: str = None):
    """Persists all ghost units for a session (full replace)."""
    conn = _get_connection(db_path)
    conn.execute("DELETE FROM ghost_units WHERE session_id = ?", (session_id,))
    for unit in ghost_store_dict.values():
        row = _unit_to_row(unit, session_id)
        conn.execute("""
            INSERT INTO ghost_units 
            (id, session_id, content, original_content, fidelity, created_at,
             last_accessed_at, access_count, cached_survival, cached_at,
             entropy_score, access_history, connections)
            VALUES (:id, :session_id, :content, :original_content, :fidelity,
                    :created_at, :last_accessed_at, :access_count, :cached_survival,
                    :cached_at, :entropy_score, :access_history, :connections)
        """, row)
    conn.commit()
    conn.close()


def load_ghosts(session_id: str, db_path: str = None) -> dict:
    """Loads all ghost units for a session. Returns {id: MemoryUnit}."""
    conn = _get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM ghost_units WHERE session_id = ?", (session_id,)
    ).fetchall()
    conn.close()
    return {row["id"]: _row_to_unit(row) for row in rows}


# ═══════════════════════════════════════════════
#  Conversation History Persistence
# ═══════════════════════════════════════════════

def save_turn(turn_data: dict, session_id: str, db_path: str = None):
    """Persists a single conversation turn."""
    conn = _get_connection(db_path)
    conn.execute("""
        INSERT INTO conversation_history
        (session_id, turn, user_input, answer, kairos_field, uncertain_claims,
         reconsolidated, memory_count, ghost_count, pressure, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        turn_data.get("turn", 0),
        turn_data.get("user_input", ""),
        turn_data.get("answer", ""),
        json.dumps(turn_data.get("kairos_field", [])),
        json.dumps(turn_data.get("uncertain_claims", [])),
        json.dumps(turn_data.get("reconsolidated", [])),
        turn_data.get("memory_count", 0),
        turn_data.get("ghost_count", 0),
        turn_data.get("pressure", "LOW"),
        time.time()
    ))
    conn.commit()
    conn.close()


def load_history(session_id: str, db_path: str = None) -> List[dict]:
    """Loads conversation history for a session."""
    conn = _get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM conversation_history WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "turn": row["turn"],
            "user_input": row["user_input"],
            "answer": row["answer"],
            "kairos_field": json.loads(row["kairos_field"] or "[]"),
            "uncertain_claims": json.loads(row["uncertain_claims"] or "[]"),
            "reconsolidated": json.loads(row["reconsolidated"] or "[]"),
            "memory_count": row["memory_count"],
            "ghost_count": row["ghost_count"],
            "pressure": row["pressure"],
        })
    return history


def clear_session(session_id: str, db_path: str = None):
    """Clears all data for a session."""
    conn = _get_connection(db_path)
    conn.execute("DELETE FROM memory_units WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM ghost_units WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# Initialize DB on import
init_db()
