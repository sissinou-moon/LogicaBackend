"""
AI Service — core reasoning engine.

Uses deepseek-r1:8b locally via Ollama.
Context injection handled by context_builder.
Intent detection handled by intent.py (cloud API).
"""

from services.intent import detect_intent
from services.context_builder import build_context
import time
import httpx
import json
import ast
import re

# Persistent httpx client — reuses TCP connection to Ollama (no reconnect overhead)
ollama_client = httpx.AsyncClient(
    base_url="http://localhost:11434",
    timeout=httpx.Timeout(300.0, connect=10.0),  # 5 min read, 10s connect
)


async def askAI(message: str, history: list = []):
    """
    Main AI reasoning function (non-streaming).
    Returns (actions, intent_result, metrics, prompt).
    """
    metrics = {}

    # 1. Intent Detection
    intent_result, intent_time = await detect_intent(message)
    metrics["intent"] = {"time": round(intent_time, 3), "called": True}

    # 2. Build Context
    knowledge_context = await build_context(intent_result, message)

    # 3. Build System Prompt
    prompt = _build_system_prompt(knowledge_context, intent_result)

    # 4. Build message list
    messages = [{"role": "system", "content": prompt}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    # 5. Call reasoning model
    reasoning_start = time.time()
    try:
        response = await ollama_client.post(
            "/api/chat",
            json={
                "model": "deepseek-r1:8b",
                "messages": messages,
                "stream": False,
                "keep_alive": "30m"
            },
        )
        response.raise_for_status()
        result = response.json()
        raw = result.get("message", {}).get("content", "").strip()
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}"
        print(f"❌ LLM ERROR: {error_detail}")
        traceback.print_exc()
        raw = f"[Error calling local AI: {error_detail}]"

    reasoning_time = time.time() - reasoning_start
    metrics["reasoning"] = {"time": round(reasoning_time, 3), "called": True}

    # Clean response
    actions = _parse_response(raw)

    return actions, intent_result, metrics, prompt


async def askAI_stream(message: str, history: list = []):
    intent_result, intent_time = await detect_intent(message)

    yield f"data: {json.dumps({'type': 'status', 'step': 'intent', 'data': intent_result})}\n\n"

    knowledge_context = await build_context(intent_result, message)
    prompt = _build_stream_prompt(knowledge_context, intent_result)

    messages = [{"role": "system", "content": prompt}]
    messages += history
    messages.append({"role": "user", "content": message})

    full_response = ""

    try:
        async with ollama_client.stream(
            "POST",
            "/api/chat",
            json={
                "model": "deepseek-r1:8b",
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "keep_alive": "30m"
            }
        ) as response:

            async for line in response.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")

                if token:
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

                if chunk.get("done"):
                    break

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


def _build_stream_prompt(knowledge_context: str, intent_result: dict) -> str:
    """Build a natural prompt for streaming — lets DeepSeek-R1 think freely using its native tags."""

    prompt = """You are an AI assistant.

You MUST respond in this exact format:

<think>
Brief reasoning steps here (very short bullet points)
</think>

Answer:
Final clear response to the user.

Rules:
- Always include <think> section
- Keep reasoning short
- Do not skip format
- Do not skip thinking section

IMPORTANT: Always output <think> section before answering.
If you do not output <think>, your response is invalid.
"""

    if knowledge_context:
        prompt += f"""
Here is relevant context from your memory:
{knowledge_context}

Use this context to inform your response.
"""

    intent = intent_result.get("intent", "question")
    intent_hints = {
        "create": "The user wants to create files or folders.",
        "modify": "The user wants to modify existing content.",
        "delete": "The user wants to delete something.",
        "search": "The user is searching for information.",
        "question": "The user is asking a question.",
        "list": "The user wants to see workspace contents.",
        "recall": "The user wants to recall past actions.",
    }

    if intent in intent_hints:
        prompt += f"\nContext: {intent_hints[intent]}\n"

    return prompt


def _build_system_prompt(knowledge_context: str, intent_result: dict) -> str:
    """Build an optimized system prompt for JSON responses (non-streaming)."""

    intent = intent_result.get("intent", "question")

    prompt = """
    You are an AI file manager.
Return ONLY valid JSON.
Keep responses short and correct.
"""

    if knowledge_context:
        prompt += f"""MEMORY CONTEXT (use this to inform your responses):
{knowledge_context}

"""

    intent_hints = {
        "create": "The user wants to CREATE something. Focus on file/folder creation.",
        "modify": "The user wants to MODIFY existing content. Use your memory to find the right file.",
        "delete": "The user wants to DELETE something. Confirm the path.",
        "search": "The user is SEARCHING for information. Use your memory context above.",
        "question": "The user is asking a QUESTION. Use your memory context to give an informed answer.",
        "list": "The user wants to LIST workspace contents.",
        "recall": "The user wants to RECALL past actions. Use the recent actions from memory.",
    }

    if intent in intent_hints:
        prompt += f"INTENT: {intent_hints[intent]}\n\n"

    prompt += """INSTRUCTIONS:
- Always respond ONLY with valid JSON.
- Use the memory context above to give informed, accurate responses.
- If the user asks about past actions, refer to the RECENT ACTIONS section.
- If the user asks to "modify" a file, use 'modify_content' action.

When generating markdown content:
- Use '-' for bullet lists, '##' for headings
- Bold important words with '**'
- Always return clean markdown content.

JSON format:
[
    {
        "action": "create_file | create_folder | delete_file | rename_file | list_files | answer | modify_content",
        "path": "file path",
        "content": "content",
        "message": "response to user"
    }
]

It can be more than one action, simply return more than one json in the list.
"""

    return prompt


def _parse_response(raw: str) -> list:
    """Parse the LLM response into a list of action dicts."""

    # Strip DeepSeek-R1 <think>...</think> blocks
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Remove markdown code blocks
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    # Fix wrapped string
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]

    # Try JSON first
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: Python literal
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass

    # Last resort: return as answer
    return [{"action": "answer", "message": raw}]