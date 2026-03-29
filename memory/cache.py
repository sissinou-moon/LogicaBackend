"""
In-memory cache for summaries, embeddings, and retrieval results.
Avoids recomputation across requests.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional


class MemoryCache:
    """Simple in-memory cache with TTL support"""

    def __init__(self, default_ttl_minutes: int = 60):
        self._cache: dict[str, dict] = {}
        self._default_ttl = timedelta(minutes=default_ttl_minutes)

    def _make_key(self, namespace: str, identifier: str) -> str:
        return f"{namespace}:{identifier}"

    def get(self, namespace: str, identifier: str) -> Optional[Any]:
        """Get a cached value. Returns None if expired or not found."""
        key = self._make_key(namespace, identifier)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # Check TTL
        if datetime.now() > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, namespace: str, identifier: str, value: Any, ttl_minutes: Optional[int] = None):
        """Cache a value with optional custom TTL."""
        key = self._make_key(namespace, identifier)
        ttl = timedelta(minutes=ttl_minutes) if ttl_minutes else self._default_ttl

        self._cache[key] = {
            "value": value,
            "expires_at": datetime.now() + ttl,
            "created_at": datetime.now()
        }

    def invalidate(self, namespace: str, identifier: str):
        """Remove a specific cached entry."""
        key = self._make_key(namespace, identifier)
        self._cache.pop(key, None)

    def invalidate_namespace(self, namespace: str):
        """Remove all entries in a namespace."""
        keys_to_remove = [k for k in self._cache if k.startswith(f"{namespace}:")]
        for key in keys_to_remove:
            del self._cache[key]

    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()


def content_hash(content: str) -> str:
    """Generate a hash of content for cache key / change detection."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# Global cache instance — shared across all services
cache = MemoryCache(default_ttl_minutes=120)
