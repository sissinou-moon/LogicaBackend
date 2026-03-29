"""
Chat Route — main conversation endpoint.

Upgraded with:
- Structured memory (semantic + episodic)
- Conditional summarization (only for meaningful content)
- Post-action reflection (only for important interactions)
- Episodic logging (always — cheap SQLite)
"""

from memory.store import add_semantic, delete_semantic
from memory.models import SemanticEntry
from services.summarizer import summarize_content, should_summarize
from services.reflection import reflect
from fastapi import APIRouter, HTTPException
from services.aiServices import askAI
from pathlib import Path
import shutil

router = APIRouter()

BASE_PATH = Path("workspace")


# ------------------ CHAT ------------------
@router.post("/")
async def chat_handler(body: dict):
    message = body.get("message")
    history = body.get("history", [])

    if not message:
        raise HTTPException(
            status_code=420,
            detail="Message is required"
        )

    # 1. AI reasoning (includes intent detection + context building)
    response, intent_result = await askAI(message, history)

    # 2. Standardize to a list of action objects
    if isinstance(response, list):
        actions = response
    elif isinstance(response, dict) and "actions" in response:
        actions = response["actions"]
    elif isinstance(response, dict):
        actions = [response]
    else:
        actions = []

    # 3. Execute actions
    for item in actions:
        if not isinstance(item, dict):
            continue

        action = item.get("action")

        if action == "create_file":
            await _handle_create_file(item)

        elif action == "create_folder":
            await _handle_create_folder(item)

        elif action == "delete_file":
            await _handle_delete_file(item)

        elif action == "rename_file":
            await _handle_rename_file(item)

        elif action == "list_files":
            await _handle_list_files(item)

        elif action == "modify_content":
            await _handle_modify_content(item)

        elif action == "answer":
            pass  # No filesystem action needed

    # 4. Post-action reflection (conditional — based on intent)
    try:
        await reflect(message, actions, intent_result)
    except Exception:
        pass  # Reflection failure should NEVER break the main flow

    # 5. Return response
    return {
        "success": True,
        "message": "Chat",
        "data": response,
        "intent": intent_result.get("intent", "unknown"),
        "updated_history": history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": str(response)}
        ]
    }


# ==================== ACTION HANDLERS ====================

async def _handle_create_file(item: dict):
    """Create a file and store in semantic memory with optional summarization."""
    path = item.get("path")
    if not path:
        item.update({"message": "Path is required to create a file"})
        return

    full_path = BASE_PATH / path
    if full_path.exists():
        item.update({"message": "File already exists"})
        return

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.touch(exist_ok=True)

    content = item.get("content", "")
    if content:
        with open(full_path, "w") as f:
            f.write(content)

    # Store in semantic memory
    # Summarize ONLY if content is worth it (avoid LLM call for tiny files)
    if content and should_summarize(content):
        summary_data = await summarize_content(content, path)
        entry = SemanticEntry(
            type="file_knowledge",
            path=path,
            summary=summary_data["summary"],
            keywords=summary_data["keywords"],
            content=content,
            importance=0.5
        )
    else:
        entry = SemanticEntry(
            type="file_knowledge",
            path=path,
            summary=f"File created at {path}",
            keywords=[],
            content=content or "",
            importance=0.3
        )

    await add_semantic(entry)


async def _handle_create_folder(item: dict):
    """Create a folder."""
    path = item.get("path")
    if not path:
        item.update({"message": "Path is required to create a folder"})
        return

    full_path = BASE_PATH / path
    if full_path.exists():
        item.update({"message": "Folder already exists"})
        return

    full_path.mkdir(parents=True, exist_ok=True)


async def _handle_delete_file(item: dict):
    """Delete a file/folder and remove from semantic memory."""
    path = item.get("path")
    if not path:
        item.update({"message": "Path is required for deletion"})
        return

    full_path = BASE_PATH / path
    if not full_path.exists():
        item.update({"message": f"Item at {path} doesn't exist"})
        return

    if full_path.is_dir():
        shutil.rmtree(full_path)
    else:
        full_path.unlink()

    # Remove from semantic memory
    await delete_semantic(path)


async def _handle_rename_file(item: dict):
    """Rename a file/folder and update semantic memory."""
    old_path = item.get("old_path")
    new_path = item.get("new_path")

    if not old_path or not new_path:
        item.update({"message": "Both old_path and new_path are required for renaming"})
        return

    full_old_path = BASE_PATH / old_path
    full_new_path = BASE_PATH / new_path

    if not full_old_path.exists():
        item.update({"message": f"Old path {old_path} doesn't exist"})
        return

    full_old_path.rename(full_new_path)

    # Update semantic memory: delete old, add new
    await delete_semantic(old_path)

    # If it's a file, re-add with new path
    if full_new_path.is_file():
        try:
            content = full_new_path.read_text()
            entry = SemanticEntry(
                type="file_knowledge",
                path=new_path,
                summary=f"Renamed from {old_path}",
                keywords=[],
                content=content,
                importance=0.4
            )
            await add_semantic(entry)
        except Exception:
            pass


async def _handle_list_files(item: dict):
    """List all files in workspace."""
    def get_all_items(path):
        items_found = []
        if not path.exists():
            return items_found
        for entry in path.iterdir():
            items_found.append(str(entry.relative_to(BASE_PATH)))
            if entry.is_dir():
                items_found.extend(get_all_items(entry))
        return items_found

    all_items = get_all_items(BASE_PATH)
    item.update({
        "message": "Workspace items: " + (", ".join(all_items) if all_items else "Empty"),
        "items": all_items
    })


async def _handle_modify_content(item: dict):
    """Modify file content and update semantic memory with optional re-summarization."""
    path = item.get("path")
    content = item.get("content")

    if not path or not content:
        item.update({"message": "Path and content are required for modifying content"})
        return

    full_path = BASE_PATH / path
    if not full_path.exists():
        item.update({"message": f"Item at {path} doesn't exist"})
        return

    with open(full_path, "w") as f:
        f.write(content)

    # Update semantic memory — re-summarize if content is substantial
    if should_summarize(content):
        summary_data = await summarize_content(content, path)
        entry = SemanticEntry(
            type="file_knowledge",
            path=path,
            summary=summary_data["summary"],
            keywords=summary_data["keywords"],
            content=content,
            importance=0.6  # Modified files get slightly higher importance
        )
    else:
        entry = SemanticEntry(
            type="file_knowledge",
            path=path,
            summary=f"Modified file at {path}",
            keywords=[],
            content=content,
            importance=0.5
        )

    await add_semantic(entry)