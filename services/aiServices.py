"""
AI Service — core reasoning engine.

Uses MiMo-V2-Flash for final reasoning (HuggingFace).
Context injection handled by context_builder.
Intent detection handled by intent.py (separate API key).
"""

from services.intent import detect_intent
from services.context_builder import build_context
import json
import ast
from openai import OpenAI

# MiMo-V2-Flash — primary reasoning model (HuggingFace)
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)


async def askAI(message: str, history: list = []):
    """
    Main AI reasoning function.

    Flow:
    1. Detect intent (lightweight, cached)
    2. Build structured context based on intent (only queries memory if needed)
    3. Inject context into system prompt
    4. Call MiMo-V2-Flash for reasoning
    5. Return structured actions + intent_result for downstream use

    Returns:
        tuple: (actions_list, intent_result_dict)
    """

    # 1. Intent Detection (OpenRouter — cheap, cached)
    intent_result = await detect_intent(message)

    # 2. Build Context (only queries memory layers that intent says are needed)
    knowledge_context = await build_context(intent_result, message)

    # 3. Build System Prompt with structured context
    prompt = _build_system_prompt(knowledge_context, intent_result)

    # 4. Build message list with history
    messages = [{"role": "system", "content": prompt}]

    # Add previous chat history (Short-term / Working memory)
    for msg in history:
        messages.append(msg)

    # Add current user message
    messages.append({"role": "user", "content": message})

    # 5. Call MiMo-V2-Flash for final reasoning
    completion = client.chat.completions.create(
        model="XiaomiMiMo/MiMo-V2-Flash:novita",
        messages=messages,
    )

    raw = completion.choices[0].message.content.strip()

    # Clean response
    actions = _parse_response(raw)

    return actions, intent_result


def _build_system_prompt(knowledge_context: str, intent_result: dict) -> str:
    """Build an optimized system prompt based on intent and available context."""

    intent = intent_result.get("intent", "question")

    # Base instruction
    prompt = """You are an AI file manager with memory and learning capabilities.

Always respond ONLY with valid JSON.

"""

    # Inject structured context (only if we have any)
    if knowledge_context:
        prompt += f"""MEMORY CONTEXT (use this to inform your responses):
{knowledge_context}

"""

    # Intent-specific hints
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