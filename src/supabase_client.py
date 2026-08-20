"""
Supabase client — authentication and chat history persistence.

Provides user auth (signup, login, logout) and CRUD operations for
persisting chat messages in Supabase PostgreSQL with Row Level Security.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from src.config import AppConfig

logger = logging.getLogger("financial_analyst.supabase")

# Module-level client cache
_supabase_client: Optional[Client] = None


def init_supabase(config: AppConfig) -> Client:
    """
    Get or create a cached Supabase client.

    Args:
        config: Application configuration with Supabase credentials.

    Returns:
        Configured Supabase client.
    """
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(config.supabase_url, config.supabase_key)
        logger.info("Supabase client initialized")
    return _supabase_client


# ══════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════

def signup_user(client: Client, email: str, password: str) -> Dict[str, Any]:
    """
    Register a new user with email and password.

    Args:
        client: Supabase client.
        email: User's email address.
        password: User's password (min 6 characters).

    Returns:
        Dict with 'success' bool and either 'user' data or 'error' message.
    """
    try:
        response = client.auth.sign_up({
            "email": email,
            "password": password,
        })

        if response.user:
            logger.info("User signed up: %s", email)
            return {
                "success": True,
                "user": response.user,
                "message": "Account created successfully! Please check your email to confirm.",
            }
        else:
            return {
                "success": False,
                "error": "Signup failed. Please try again.",
            }

    except Exception as e:
        error_msg = str(e)
        logger.error("Signup failed for %s: %s", email, error_msg)

        # Parse common Supabase auth errors
        if "already registered" in error_msg.lower() or "already been registered" in error_msg.lower():
            return {"success": False, "error": "This email is already registered. Please log in instead."}
        if "password" in error_msg.lower() and "short" in error_msg.lower():
            return {"success": False, "error": "Password must be at least 6 characters."}
        if "valid email" in error_msg.lower() or "invalid" in error_msg.lower():
            return {"success": False, "error": "Please enter a valid email address."}

        return {"success": False, "error": f"Signup failed: {error_msg}"}


def login_user(client: Client, email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user with email and password.

    Args:
        client: Supabase client.
        email: User's email address.
        password: User's password.

    Returns:
        Dict with 'success' bool and either 'user'/'session' data or 'error' message.
    """
    try:
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password,
        })

        if response.user and response.session:
            logger.info("User logged in: %s", email)
            return {
                "success": True,
                "user": response.user,
                "session": response.session,
            }
        else:
            return {
                "success": False,
                "error": "Login failed. Please check your credentials.",
            }

    except Exception as e:
        error_msg = str(e)
        logger.error("Login failed for %s: %s", email, error_msg)

        if "invalid" in error_msg.lower() and "credentials" in error_msg.lower():
            return {"success": False, "error": "Invalid email or password."}
        if "email not confirmed" in error_msg.lower():
            return {"success": False, "error": "Please confirm your email before logging in."}

        return {"success": False, "error": f"Login failed: {error_msg}"}


def logout_user(client: Client) -> bool:
    """
    Sign out the current user.

    Returns:
        True if logout succeeded.
    """
    try:
        client.auth.sign_out()
        logger.info("User logged out")
        return True
    except Exception as e:
        logger.error("Logout failed: %s", e)
        return False


def get_current_user(client: Client) -> Optional[Any]:
    """
    Get the currently authenticated user, if any.

    Returns:
        User object or None.
    """
    try:
        response = client.auth.get_user()
        return response.user if response else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════
# CHAT HISTORY PERSISTENCE
# ══════════════════════════════════════════════════════

def save_message(
    client: Client,
    user_id: str,
    role: str,
    content: str,
    badge: Optional[str] = None,
    route: Optional[str] = None,
) -> bool:
    """
    Persist a single chat message to Supabase.

    Args:
        client: Supabase client.
        user_id: The authenticated user's UUID.
        role: 'user' or 'assistant'.
        content: The message text.
        badge: Optional HTML badge string.
        route: Optional route label (csv, pdf, chart, general).

    Returns:
        True if saved successfully.
    """
    try:
        client.table("chat_history").insert({
            "user_id": user_id,
            "role": role,
            "content": content,
            "badge": badge,
            "route": route,
        }).execute()
        return True
    except Exception as e:
        logger.error("Failed to save message: %s", e)
        return False


def load_chat_history(
    client: Client,
    user_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Load chat history for a user from Supabase.

    Args:
        client: Supabase client.
        user_id: The authenticated user's UUID.
        limit: Maximum number of messages to retrieve.

    Returns:
        List of message dicts with role, content, badge keys.
        Messages are ordered oldest-first for display.
    """
    try:
        response = (
            client.table("chat_history")
            .select("role, content, badge, route, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        messages = []
        for row in response.data or []:
            messages.append({
                "role": row["role"],
                "content": row["content"],
                "chart": None,  # Charts cannot be serialized to DB
                "badge": row.get("badge"),
            })
        logger.info("Loaded %d messages for user %s", len(messages), user_id)
        return messages

    except Exception as e:
        logger.error("Failed to load chat history: %s", e)
        return []


def clear_chat_history(client: Client, user_id: str) -> bool:
    """
    Delete all chat messages for a user.

    Args:
        client: Supabase client.
        user_id: The authenticated user's UUID.

    Returns:
        True if cleared successfully.
    """
    try:
        client.table("chat_history").delete().eq("user_id", user_id).execute()
        logger.info("Cleared chat history for user %s", user_id)
        return True
    except Exception as e:
        logger.error("Failed to clear chat history: %s", e)
        return False
