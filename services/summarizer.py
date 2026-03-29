"""
Auto-Summarization — transforms raw file content into structured knowledge.
Uses OpenRouter (dedicated API key) for summarization.

NOT called for every file — only when:
  - File has meaningful content (not empty/tiny)
  - Summary isn't already cached
  - Content has actually changed
"""

import json
from openai import OpenAI
from memory.cache import cache, content_hash

# Dedicated OpenRouter client for summarization
summary_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Minimum content length worth summarizing
MIN_CONTENT_LENGTH = 50


async def summarize_content(content: str, path: str) -> dict:
    """
    Generate a structured summary of file content.

    Returns:
        {
            "summary": "2-line description of what this file contains",
            "keywords": ["keyword1", "keyword2", ...]
        }

    If content is too short or summarization fails, returns a simple fallback.
    """
    # Skip tiny content — not worth an API call
    if len(content.strip()) < MIN_CONTENT_LENGTH:
        return {
            "summary": f"Short file at {path}",
            "keywords": _extract_simple_keywords(content, path)
        }

    # Check cache — same content = same summary
    c_hash = content_hash(content)
    cache_key = f"{path}:{c_hash}"
    cached = cache.get("summary", cache_key)
    if cached is not None:
        return cached

    prompt = """Analyze this file content and return a JSON object with:
1. "summary": A concise 2-line summary of what this file contains and its purpose.
2. "keywords": A list of 3-7 relevant keywords/topics.

Return ONLY valid JSON, no other text.

Example:
{"summary": "Configuration file for database connection settings. Contains host, port, and credentials for PostgreSQL.", "keywords": ["database", "config", "postgresql", "connection"]}
"""

    try:
        completion = summary_client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"File path: {path}\n\nContent:\n{content[:2000]}"}
            ],
            max_tokens=200,
            temperature=0.0,
        )

        raw = completion.choices[0].message.content.strip()

        # Clean markdown wrapping
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)

        # Validate structure
        if "summary" not in result:
            result["summary"] = f"File at {path}"
        if "keywords" not in result or not isinstance(result["keywords"], list):
            result["keywords"] = _extract_simple_keywords(content, path)

    except Exception:
        # Fallback — don't break the flow
        result = {
            "summary": f"File at {path} ({len(content)} chars)",
            "keywords": _extract_simple_keywords(content, path)
        }

    # Cache for 2 hours — summaries rarely change for same content
    cache.set("summary", cache_key, result, ttl_minutes=120)

    return result


def _extract_simple_keywords(content: str, path: str) -> list[str]:
    """
    Cheap fallback: extract keywords from path and content without calling LLM.
    """
    keywords = []

    # From path
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        name = part.split(".")[0]
        if name and len(name) > 2:
            keywords.append(name.lower())

    # From content — most frequent meaningful words
    words = content.lower().split()
    word_freq = {}
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
                  "of", "and", "or", "but", "not", "with", "from", "by", "as", "it", "this",
                  "that", "be", "has", "have", "had", "do", "does", "did", "will", "would",
                  "could", "should", "can", "may", "might", "if", "then", "else"}

    for word in words:
        cleaned = ''.join(c for c in word if c.isalnum())
        if cleaned and len(cleaned) > 2 and cleaned not in stop_words:
            word_freq[cleaned] = word_freq.get(cleaned, 0) + 1

    # Top 5 by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords.extend([w for w, _ in sorted_words[:5]])

    return list(set(keywords))[:7]


def should_summarize(content: str) -> bool:
    """
    Determine if content is worth summarizing with an LLM call.
    Returns False for empty/trivial content.
    """
    if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
        return False
    return True
