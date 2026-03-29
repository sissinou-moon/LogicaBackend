"""
Context Builder — replaces raw "File: path\nContent: ..." injection.
Builds structured, LLM-optimized context from semantic + episodic memory.
"""

from memory.store import search_semantic, search_episodes, get_recent_episodes, reinforce


async def build_context(intent_result: dict, query: str) -> str:
    """
    Based on intent, selects which memory layers to query,
    then formats results in a structured, LLM-friendly way.

    Args:
        intent_result: dict from detect_intent() with needs_semantic/needs_episodic flags
        query: the user's message

    Returns:
        Formatted context string ready for prompt injection
    """
    sections = []

    # 1. Semantic Memory — knowledge retrieval
    if intent_result.get("needs_semantic"):
        semantic_results = await search_semantic(query, n=5, min_score=0.3)

        if semantic_results:
            # Reinforce importance of accessed memories (non-blocking, best-effort)
            for result in semantic_results:
                if result.get("path"):
                    await reinforce(result["path"], boost=0.05)

            knowledge_block = _format_semantic(semantic_results)
            sections.append(knowledge_block)

    # 2. Episodic Memory — past actions
    if intent_result.get("needs_episodic"):
        # For "recall" intent, search by query; otherwise get recent
        if intent_result.get("intent") == "recall":
            episodes = await search_episodes(query, limit=8)
        else:
            episodes = await get_recent_episodes(limit=5)

        if episodes:
            episodes_block = _format_episodes(episodes)
            sections.append(episodes_block)

    if not sections:
        return ""

    return "\n\n".join(sections)


def _format_semantic(results: list[dict]) -> str:
    """Format semantic memory results into structured context."""
    lines = ["=== RELEVANT KNOWLEDGE ==="]

    for i, r in enumerate(results, 1):
        path = r.get("path", "unknown")
        summary = r.get("summary", "")
        keywords = r.get("keywords", [])
        importance = r.get("importance", 0.5)
        similarity = r.get("similarity", 0)

        # Importance stars (visual indicator for the LLM)
        stars = "★" * int(importance * 5) + "☆" * (5 - int(importance * 5))

        entry = f"{i}. 📄 File: {path}"
        if summary:
            entry += f"\n   Summary: {summary}"
        if keywords:
            entry += f"\n   Topics: {', '.join(keywords[:5])}"
        entry += f"\n   Relevance: {stars} ({similarity:.0%})"

        lines.append(entry)

    lines.append("=" * 30)
    return "\n".join(lines)


def _format_episodes(episodes: list[dict]) -> str:
    """Format episodic memory into a timeline."""
    lines = ["=== RECENT ACTIONS ==="]

    for ep in episodes:
        timestamp = ep.get("timestamp", "")
        # Shorten timestamp to readable format
        try:
            ts_short = timestamp[:16].replace("T", " ")
        except Exception:
            ts_short = timestamp

        event = ep.get("event", "")
        summary = ep.get("summary", "")
        path = ep.get("path", "")

        event_icon = {
            "created_file": "📝",
            "modified_file": "✏️",
            "deleted_file": "🗑️",
            "renamed_file": "📋",
            "searched": "🔍",
            "answered": "💬",
            "reflected": "🧠",
        }.get(event, "📌")

        line = f"- [{ts_short}] {event_icon} {event}"
        if path:
            line += f" → {path}"
        if summary:
            line += f" | {summary}"

        lines.append(line)

    lines.append("=" * 30)
    return "\n".join(lines)
