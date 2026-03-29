"""
Central Memory Store — the brain of the agent.

Semantic Memory: ChromaDB (cloud) — knowledge, summaries, structured data
Episodic Memory: SQLite (local) — event logs, action history
"""

import chromadb
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional
from memory.models import SemanticEntry, EpisodicEntry
from memory.cache import cache, content_hash


# ==================== CHROMADB SETUP ====================

chroma_client = chromadb.CloudClient(
    api_key='ck-t3nTNaVrLJsyZPgFE1X4ZeXg492d6fc9dMV2ATtFnLG',
    tenant='f05fcb00-3bec-4eaf-82a3-915dd5a86bfd',
    database='pfp'
)

semantic_collection = chroma_client.get_or_create_collection(name="semantic_knowledge")


# ==================== SQLITE SETUP ====================

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "episodic_memory.db")

def _init_sqlite():
    """Initialize SQLite database for episodic memory."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            path TEXT DEFAULT '',
            timestamp TEXT NOT NULL,
            summary TEXT DEFAULT '',
            user_message TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_episodes_event ON episodes(event)
    """)
    conn.commit()
    conn.close()

# Initialize on import
_init_sqlite()


# ==================== SEMANTIC MEMORY (ChromaDB) ====================

async def add_semantic(entry: SemanticEntry):
    """
    Store structured knowledge in semantic memory.
    If entry with same path exists, update it instead.
    """
    doc_id = entry.path if entry.path else content_hash(entry.content)

    metadata = {
        "type": entry.type,
        "path": entry.path,
        "summary": entry.summary,
        "keywords": json.dumps(entry.keywords),
        "importance": entry.importance,
        "updated_at": datetime.now().isoformat()
    }

    # Build the document text: summary + content for better semantic search
    document = f"{entry.summary}\n\n{entry.content}" if entry.summary else entry.content

    try:
        # Try to get existing — if it exists, update
        existing = semantic_collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            semantic_collection.update(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
        else:
            semantic_collection.add(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
    except Exception:
        # If get fails, just add
        semantic_collection.add(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata]
        )

    # Invalidate relevant caches
    cache.invalidate("search", content_hash(entry.path))


async def search_semantic(query: str, n: int = 5, min_score: float = 0.3) -> list[dict]:
    """
    Search semantic memory with score filtering.
    Returns structured results sorted by relevance.
    """
    # Check cache first
    cache_key = content_hash(f"{query}:{n}")
    cached = cache.get("search", cache_key)
    if cached is not None:
        return cached

    try:
        results = semantic_collection.query(
            query_texts=[query],
            n_results=min(n, 10),  # cap at 10
            include=["documents", "metadatas", "distances"]
        )
    except Exception:
        return []

    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    structured_results = []
    for i in range(len(docs)):
        # ChromaDB returns L2 distance — lower = more similar
        # Convert to similarity score (0-1 range, higher = better)
        similarity = max(0, 1 - (distances[i] / 2))

        if similarity < min_score:
            continue

        meta = metas[i] if i < len(metas) else {}

        # Parse keywords back from JSON string
        keywords = []
        keywords_raw = meta.get("keywords", "[]")
        try:
            keywords = json.loads(keywords_raw) if isinstance(keywords_raw, str) else keywords_raw
        except (json.JSONDecodeError, TypeError):
            keywords = []

        structured_results.append({
            "path": meta.get("path", ""),
            "type": meta.get("type", "file_knowledge"),
            "summary": meta.get("summary", ""),
            "keywords": keywords,
            "content": docs[i],
            "importance": meta.get("importance", 0.5),
            "similarity": round(similarity, 3)
        })

    # Sort by: similarity * importance (weighted relevance)
    structured_results.sort(
        key=lambda x: x["similarity"] * (0.7 + 0.3 * x["importance"]),
        reverse=True
    )

    # Cache the results (5 min TTL for search results)
    cache.set("search", cache_key, structured_results, ttl_minutes=5)

    return structured_results


async def reinforce(path: str, boost: float = 0.1):
    """
    Increase importance of a memory when it's accessed/used.
    Capped at 1.0.
    """
    try:
        result = semantic_collection.get(ids=[path])
        if result and result["ids"]:
            meta = result["metadatas"][0]
            current_importance = meta.get("importance", 0.5)
            new_importance = min(1.0, current_importance + boost)
            meta["importance"] = new_importance
            meta["updated_at"] = datetime.now().isoformat()
            semantic_collection.update(
                ids=[path],
                metadatas=[meta]
            )
    except Exception:
        pass  # Non-critical — don't break flow


async def delete_semantic(path: str):
    """Remove a semantic memory entry by path."""
    try:
        semantic_collection.delete(ids=[path])
        cache.invalidate("search", content_hash(path))
    except Exception:
        pass


# ==================== EPISODIC MEMORY (SQLite) ====================

async def add_episode(entry: EpisodicEntry):
    """Log an event in episodic memory."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO episodes (event, path, timestamp, summary, user_message) VALUES (?, ?, ?, ?, ?)",
        (entry.event, entry.path, entry.timestamp, entry.summary, entry.user_message)
    )
    conn.commit()
    conn.close()


async def get_recent_episodes(limit: int = 10) -> list[dict]:
    """Get the most recent episodes."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


async def search_episodes(query: str, limit: int = 5) -> list[dict]:
    """
    Search episodic memory by keyword match in summary and user_message.
    (SQLite FTS would be better for production, but LIKE works for now)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Split query into words for flexible matching
    words = query.lower().split()
    conditions = []
    params = []
    for word in words[:5]:  # cap at 5 words to avoid huge queries
        conditions.append("(LOWER(summary) LIKE ? OR LOWER(user_message) LIKE ? OR LOWER(path) LIKE ?)")
        params.extend([f"%{word}%", f"%{word}%", f"%{word}%"])

    where_clause = " OR ".join(conditions) if conditions else "1=1"

    cursor = conn.execute(
        f"SELECT * FROM episodes WHERE {where_clause} ORDER BY timestamp DESC LIMIT ?",
        params + [limit]
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


async def prune_old_episodes(days: int = 30):
    """Remove episodic entries older than the specified number of days."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM episodes WHERE timestamp < ?", (cutoff,))
    conn.commit()
    conn.close()


# ==================== LIFECYCLE ====================

async def startup_maintenance():
    """Run on app startup: prune old episodes."""
    await prune_old_episodes(days=30)
