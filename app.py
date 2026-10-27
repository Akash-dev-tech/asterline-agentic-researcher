"""FastAPI production web backend for Asterline Labs Multi-Agent Assistant.

Serves:
- Authentication & JWT session handling (/api/auth)
- Multi-session chat management & persistence (/api/sessions)
- Isolated LangGraph multi-agent turns (/api/chat)
- User workspace memory management (/api/memory)
- Interactive SPA frontend static assets
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents import run_assistant_turn
from src.auth import create_access_token, get_current_user, hash_password, verify_password
from src.memory_db import (
    clear_all_user_memory,
    clear_user_chat_history,
    create_chat_session,
    create_user,
    delete_chat_session,
    delete_memory_key,
    get_chat_history,
    get_chat_session_messages,
    get_user_by_username,
    get_user_chat_sessions,
    get_user_memories,
    init_db,
    update_session_timestamp,
)

load_dotenv()

# Initialize SQLite database on module import
init_db()

app = FastAPI(
    title="Asterline Labs Assistant API",
    description="Production backend for Asterline multi-agent research and memory assistant.",
    version="2.1.0",
)

# Enable CORS for decoupled frontends or local tooling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------
class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    retrieval_mode: str = Field(default="hybrid", pattern="^(hybrid|vector_only)$")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    warnings: List[str]
    search_queries: List[str]
    retrieved_chunks: List[str]
    memory_context: Dict[str, str]
    memory_actions: List[str]
    retry_count: int
    verifier_feedback: str


# ----------------------------------------------------------------------
# Authentication Routes
# ----------------------------------------------------------------------
@app.post("/api/auth/register", response_model=AuthResponse)
def register(req: AuthRequest) -> AuthResponse:
    """Register a new employee/user account."""
    clean_username = req.username.strip().lower()
    existing = get_user_by_username(clean_username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists. Please sign in instead.",
        )

    hashed = hash_password(req.password)
    success = create_user(clean_username, hashed)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user account.",
        )

    token = create_access_token({"sub": clean_username})
    return AuthResponse(access_token=token, username=clean_username)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: AuthRequest) -> AuthResponse:
    """Authenticate existing credentials and issue a signed JWT token."""
    clean_username = req.username.strip().lower()
    user = get_user_by_username(clean_username)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": clean_username})
    return AuthResponse(access_token=token, username=clean_username)


@app.get("/api/auth/me")
def get_current_user_profile(user_id: str = Depends(get_current_user)) -> Dict[str, str]:
    """Retrieve identity of the actively authenticated user."""
    return {"username": user_id}


# ----------------------------------------------------------------------
# Multi-Session Chat Routes (ChatGPT-style session management)
# ----------------------------------------------------------------------
@app.get("/api/sessions")
def list_sessions(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """List all saved chat sessions for the logged-in user in chronological order."""
    sessions = get_user_chat_sessions(user_id)
    return {"sessions": sessions}


@app.post("/api/sessions")
def create_new_session(
    req: CreateSessionRequest = CreateSessionRequest(),
    user_id: str = Depends(get_current_user),
) -> Dict[str, str]:
    """Create a new distinct chat session."""
    session_id = uuid.uuid4().hex
    title = (req.title or "New Chat").strip()
    create_chat_session(session_id, user_id, title)
    return {"session_id": session_id, "title": title}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieve message history for a specific conversation session."""
    messages = get_chat_session_messages(user_id, session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/api/sessions/{session_id}")
def remove_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete an individual chat session and its turns."""
    delete_chat_session(user_id, session_id)
    return {"status": "deleted", "session_id": session_id}


# ----------------------------------------------------------------------
# Core Chat Execution Route
# ----------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat_turn(req: ChatRequest, user_id: str = Depends(get_current_user)) -> ChatResponse:
    """Execute a multi-agent RAG turn within a specific session, preserving global user memory."""
    try:
        session_id = (req.session_id or "").strip()
        # If no session specified, auto-create a session with title derived from user question
        if not session_id:
            session_id = uuid.uuid4().hex
            title = req.message.strip().split("\n")[0].strip() or "New Chat"
            if len(title) > 36:
                title = title[:33].rstrip() + "..."
            create_chat_session(session_id, user_id, title)
        else:
            existing_ids = {s["session_id"] for s in get_user_chat_sessions(user_id)}
            if session_id not in existing_ids:
                title = req.message.strip().split("\n")[0].strip() or "New Chat"
                if len(title) > 36:
                    title = title[:33].rstrip() + "..."
                create_chat_session(session_id, user_id, title)
            else:
                update_session_timestamp(session_id)

        result_state = run_assistant_turn(
            user_id=user_id,
            user_message=req.message,
            retrieval_mode=req.retrieval_mode,
            session_id=session_id,
        )
        update_session_timestamp(session_id)

        return ChatResponse(
            answer=result_state["final_answer"],
            session_id=session_id,
            warnings=result_state.get("warnings") or result_state.get("guardrail_warnings", []),
            search_queries=result_state.get("search_queries", []),
            retrieved_chunks=result_state.get("retrieved_chunks", []),
            memory_context=result_state.get("memory_context") or result_state.get("user_memory", {}),
            memory_actions=result_state.get("memory_actions_taken", []),
            retry_count=result_state.get("retry_count", 0),
            verifier_feedback=result_state.get("verifier_feedback", ""),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agent workflow: {e}",
        )


# ----------------------------------------------------------------------
# Global User Memory (Cabinet 2) Management Routes
# ----------------------------------------------------------------------
@app.get("/api/memory")
def get_memory(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Fetch stored key-value workspace attributes for the authenticated user."""
    memories = get_user_memories(user_id)
    return {"user_id": user_id, "memory": memories}


@app.delete("/api/memory")
def delete_memory(
    key: Optional[str] = None,
    clear_history: bool = False,
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Clear all memory attributes or surgically delete a single slot for the user."""
    if key:
        delete_memory_key(user_id, key)
        action_msg = f"Deleted memory slot '{key}'"
    else:
        clear_all_user_memory(user_id)
        action_msg = "Cleared all user memory attributes"

    if clear_history:
        clear_user_chat_history(user_id)
        action_msg += " and cleared chat history"

    updated_memories = get_user_memories(user_id)
    return {
        "user_id": user_id,
        "message": action_msg,
        "memory": updated_memories,
    }


# ----------------------------------------------------------------------
# Frontend SPA Static Delivery
# ----------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_spa():
    """Serve the single-page application entry point."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return {"message": "Asterline Assistant API is running. UI index.html not yet initialized."}
    return FileResponse(index_file)


# ----------------------------------------------------------------------
# Direct Execution Server Runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
