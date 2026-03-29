"""
Reflection Engine — post-action learning loop.

CONDITIONAL: Only runs for important interactions:
  ✅ Code writing / modification
  ✅ Important decisions / complex questions
  ✅ High-accuracy scenarios
  ❌ Simple file creation (skip)
  ❌ Basic list/delete operations (skip)
"""

from datetime import datetime
from memory.store import add_episode, add_semantic
from memory.models import SemanticEntry, EpisodicEntry


async def reflect(user_message: str, actions: list, intent_result: dict):
    """
    Post-action reflection. Does two things:
    1. ALWAYS: Log episodic memory (cheap — just SQLite insert)
    2. CONDITIONALLY: Extract and store new knowledge (expensive — skip for trivial ops)

    Args:
        user_message: original user message
        actions: list of action dicts that were executed
        intent_result: dict from detect_intent()
    """
    if not actions:
        return

    # === STEP 1: Always log episodes (cheap, SQLite) ===
    await _log_episodes(user_message, actions)

    # === STEP 2: Conditional reflection (only for important interactions) ===
    if intent_result.get("needs_reflection", False):
        await _extract_knowledge(user_message, actions)


async def _log_episodes(user_message: str, actions: list):
    """
    Log each action as an episodic event.
    This is ALWAYS done — it's just a SQLite insert.
    """
    timestamp = datetime.now().isoformat()

    for item in actions:
        if not isinstance(item, dict):
            continue

        action = item.get("action", "")
        path = item.get("path", "")
        message = item.get("message", "")

        # Map action to event type
        event_map = {
            "create_file": "created_file",
            "create_folder": "created_file",
            "modify_content": "modified_file",
            "delete_file": "deleted_file",
            "rename_file": "renamed_file",
            "list_files": "searched",
            "answer": "answered",
        }

        event = event_map.get(action, action)

        # Build a human-readable summary
        summary = _build_action_summary(action, path, message)

        episode = EpisodicEntry(
            event=event,
            path=path,
            timestamp=timestamp,
            summary=summary,
            user_message=user_message[:500]  # cap to avoid huge entries
        )

        await add_episode(episode)


async def _extract_knowledge(user_message: str, actions: list):
    """
    For important interactions (modify, search, question),
    extract any new knowledge worth remembering.

    This does NOT call the LLM — it extracts structured data
    from the actions themselves. Much cheaper than a full reflection call.
    """
    for item in actions:
        if not isinstance(item, dict):
            continue

        action = item.get("action", "")
        path = item.get("path", "")
        content = item.get("content", "")
        message = item.get("message", "")

        # For answer actions with substantial content — store the Q&A pair
        if action == "answer" and message and len(message) > 100:
            entry = SemanticEntry(
                type="qa_knowledge",
                path=f"qa/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                summary=f"Q: {user_message[:100]} → A: {message[:100]}",
                keywords=_extract_words(user_message),
                content=f"Question: {user_message}\nAnswer: {message}",
                importance=0.6
            )
            await add_semantic(entry)


def _build_action_summary(action: str, path: str, message: str) -> str:
    """Build a human-readable summary of an action."""
    summaries = {
        "create_file": f"Created file: {path}",
        "create_folder": f"Created folder: {path}",
        "modify_content": f"Modified content of: {path}",
        "delete_file": f"Deleted: {path}",
        "rename_file": f"Renamed file at: {path}",
        "list_files": "Listed workspace contents",
        "answer": f"Answered: {message[:80]}..." if message else "Provided an answer",
    }
    return summaries.get(action, f"{action}: {path}")


def _extract_words(text: str) -> list[str]:
    """Extract meaningful words from text for keywords."""
    stop_words = {"the", "a", "an", "is", "are", "was", "in", "on", "at", "to", "for",
                  "of", "and", "or", "but", "not", "with", "from", "by", "as", "it",
                  "this", "that", "be", "do", "can", "i", "you", "my", "me", "what",
                  "how", "where", "when", "why", "please", "want", "need", "make",
                  "create", "file", "folder"}

    words = text.lower().split()
    meaningful = [w for w in words if w.isalpha() and len(w) > 2 and w not in stop_words]
    return list(set(meaningful))[:7]
