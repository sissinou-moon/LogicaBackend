"""
Intent Detection — classifies user messages BEFORE retrieval.
Uses OpenRouter (dedicated API key) for lightweight classification.
Determines whether memory lookup is even needed.
"""

import time
from openai import OpenAI
from memory.cache import cache, content_hash

# Dedicated OpenRouter client for intent detection
intent_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Intent categories and whether they need memory retrieval
INTENT_CONFIG = {
    "create": {"needs_semantic": False, "needs_episodic": False},
    "modify": {"needs_semantic": True, "needs_episodic": False},
    "delete": {"needs_semantic": False, "needs_episodic": False},
    "search": {"needs_semantic": True, "needs_episodic": True},
    "question": {"needs_semantic": True, "needs_episodic": True},
    "list": {"needs_semantic": False, "needs_episodic": False},
    "recall": {"needs_semantic": False, "needs_episodic": True},  # "what did I do?"
}

# Whether reflection is warranted for this intent
REFLECTION_INTENTS = {"modify", "search", "question"}


async def detect_intent(message: str) -> tuple[dict, float]:
    """
    Classify user message into an intent.
    Returns (result_dict, time_taken).
    """
    start_time = time.time()
    # Check cache — same message = same intent
    cache_key = content_hash(message)
    cached = cache.get("intent", cache_key)
    if cached is not None:
        return cached, time.time() - start_time

    prompt = """Classify this user message into exactly ONE intent. 
Reply with ONLY the intent word, nothing else.

Intents:
- create → user wants to create files or folders
- modify → user wants to edit/update existing file content
- delete → user wants to remove/delete files or folders
- search → user is looking for information about their files or content
- question → user is asking a general question or wants explanation
- list → user wants to see workspace structure or file listing
- recall → user asks about past actions ("what did I do?", "what happened?")

User message: """

    try:
        completion = intent_client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=10,
            temperature=0.0,
        )

        raw_intent = completion.choices[0].message.content.strip().lower()

        # Normalize — handle edge cases
        intent = raw_intent if raw_intent in INTENT_CONFIG else "question"

    except Exception:
        # Fallback: default to "question" (safest — will retrieve memory)
        intent = "question"

    config = INTENT_CONFIG[intent]

    result = {
        "intent": intent,
        "needs_semantic": config["needs_semantic"],
        "needs_episodic": config["needs_episodic"],
        "needs_reflection": intent in REFLECTION_INTENTS,
    }

    # Cache for 30 min — same message rarely changes intent
    cache.set("intent", cache_key, result, ttl_minutes=30)

    time_taken = time.time() - start_time
    return result, time_taken
