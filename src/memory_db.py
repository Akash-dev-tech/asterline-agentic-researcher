"""Persistent user memory, authentication credentials, and multi-session chat history using SQLite.

Manages:
- Cabinet 2: Global user-scoped workspace attributes (persisting across all sessions)
- Multi-session chat management: chat_sessions & session-partitioned chat_history
- Authentication accounts with hashed passwords
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "memory.db"
DB_PATH = str(DEFAULT_DB_PATH)

# Guardrail: Forbidden attributes that must never be persisted to SQLite
FORBIDDEN_KEYS = frozenset({
    "token",
    "api_key",
    "ingestion_key",
    "health",
    "medical",
    "secret",
    "password",
    "surgery",
    "prescription",
    "diagnosis",
})


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize the SQLite database with required tables and columns."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id TEXT NOT NULL,
                attribute_key TEXT NOT NULL,
                attribute_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, attribute_key)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Automatic schema migration: ensure session_id column exists
        cursor.execute("PRAGMA table_info(chat_history)")
        columns = [row[1] for row in cursor.fetchall()]
        if "session_id" not in columns:
            cursor.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")

        conn.commit()


# ----------------------------------------------------------------------
# User Authentication Helpers
# ----------------------------------------------------------------------
def create_user(username: str, hashed_password: str, db_path: str = DB_PATH) -> bool:
    """Create a new user account with hashed password."""
    clean_username = username.strip().lower()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                (clean_username, hashed_password),
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False


def get_user_by_username(username: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch user account credentials by username."""
    clean_username = username.strip().lower()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, hashed_password, created_at FROM users WHERE username = ?",
            (clean_username,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "username": row[1],
            "hashed_password": row[2],
            "created_at": row[3],
        }


# ----------------------------------------------------------------------
# User Memory (Cabinet 2) Operations - Global to User
# ----------------------------------------------------------------------
def upsert_memory(user_id: str, key: str, value: str, db_path: str = DB_PATH) -> bool:
    """Insert or update a user memory attribute under strict security guardrails.

    Persists globally per user_id so attributes span across all chat sessions.
    """
    clean_key = key.strip().lower().replace(" ", "_")

    if (
        clean_key in FORBIDDEN_KEYS
        or any(forbidden in clean_key for forbidden in FORBIDDEN_KEYS)
        or "[REDACTED_SECRET]" in value
    ):
        return False

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_memories (user_id, attribute_key, attribute_value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, attribute_key)
            DO UPDATE SET attribute_value = excluded.attribute_value,
                          updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, clean_key, value.strip()),
        )
        conn.commit()
    return True


def get_user_memories(user_id: str, db_path: str = DB_PATH) -> Dict[str, str]:
    """Retrieve all memory attributes for a specific user."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT attribute_key, attribute_value
            FROM user_memories
            WHERE user_id = ?
            ORDER BY attribute_key
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
    return {row[0]: row[1] for row in rows}


def delete_memory_key(user_id: str, key: str, db_path: str = DB_PATH) -> None:
    """Surgically delete a single memory attribute for a specific user."""
    clean_key = key.strip().lower().replace(" ", "_")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_memories WHERE user_id = ? AND attribute_key = ?",
            (user_id, clean_key),
        )
        conn.commit()


def clear_all_user_memory(user_id: str, db_path: str = DB_PATH) -> None:
    """Clear all memory attributes for a specific user across all sessions."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
        conn.commit()


# ----------------------------------------------------------------------
# Multi-Session Chat Management Operations
# ----------------------------------------------------------------------
def create_chat_session(
    session_id: str,
    user_id: str,
    title: str = "New Chat",
    db_path: str = DB_PATH,
) -> bool:
    """Create a new chat session for a user."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_sessions (session_id, user_id, title, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET title = excluded.title, updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, user_id, title),
        )
        conn.commit()
    return True


def get_user_chat_sessions(user_id: str, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all chat sessions for a user ordered by most recently updated."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT session_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC, created_at DESC
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
    return [
        {
            "session_id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
        }
        for row in rows
    ]


def get_chat_session_messages(
    user_id: str,
    session_id: str,
    limit: int = 100,
    db_path: str = DB_PATH,
) -> List[Dict[str, str]]:
    """Retrieve all messages for a specific session belonging to the user."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role, content
            FROM chat_history
            WHERE user_id = ? AND session_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (user_id, session_id, limit),
        )
        rows = cursor.fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def delete_chat_session(user_id: str, session_id: str, db_path: str = DB_PATH) -> bool:
    """Delete a specific chat session and all its associated messages."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM chat_history WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        cursor.execute(
            "DELETE FROM chat_sessions WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        conn.commit()
    return True


def update_session_timestamp(session_id: str, db_path: str = DB_PATH) -> None:
    """Update the updated_at timestamp of a chat session."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()


# ----------------------------------------------------------------------
# Chat Message Ingestion & Turn History
# ----------------------------------------------------------------------
def add_chat_message(
    user_id: str,
    role: str,
    content: str,
    session_id: Optional[str] = None,
    db_path: str = DB_PATH,
) -> None:
    """Append a message turn to the user's chat history, optionally tied to a session."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (session_id, user_id, role, content),
        )
        conn.commit()


def get_chat_history(
    user_id: str,
    limit: int = 10,
    session_id: Optional[str] = None,
    db_path: str = DB_PATH,
) -> List[Dict[str, str]]:
    """Retrieve recent conversation turns for a user, optionally filtered by session."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if session_id:
            query = """
                SELECT role, content FROM (
                    SELECT id, role, content FROM chat_history
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
            """
            params = (user_id, session_id, limit)
        else:
            query = """
                SELECT role, content FROM (
                    SELECT id, role, content FROM chat_history
                    WHERE user_id = ?
                    ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
            """
            params = (user_id, limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def clear_user_chat_history(user_id: str, db_path: str = DB_PATH) -> None:
    """Clear all chat conversation turns and sessions for a user."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))
        conn.commit()
