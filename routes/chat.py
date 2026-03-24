from fastapi import APIRouter, HTTPException
from services.aiServices import askAI
from pathlib import Path
import shutil

router = APIRouter()

BASE_PATH = Path("workspace")

# ------------------ CHAT ------------------
@router.post("/")
async def function(body: dict):
    message = body.get("message")

    if not message :
        raise HTTPException(
            status_code= 420,
            detail= "Message is required"
        )


    response = await askAI(message)

    # Standardize to a list of action objects
    if isinstance(response, list):
        actions = response
    elif isinstance(response, dict) and "actions" in response:
        actions = response["actions"]
    elif isinstance(response, dict):
        actions = [response]
    else:
        actions = []

    for item in actions:
        if not isinstance(item, dict):
            continue
            
        action = item.get("action")
        if action == "create_file":
            path = item.get("path")
            if path:
                full_path = BASE_PATH / path
                if full_path.exists():
                    item.update({
                        "message": "File already exists"
                    })
                else:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.touch(exist_ok=True)
                    if item.get("content"):
                        with open(full_path, "w") as f:
                            f.write(item.get("content"))
            else:
                item.update({"message": "Path is required to create a file"})
        elif action == "create_folder":
            path = item.get("path")
            if path:
                full_path = BASE_PATH / path
                if full_path.exists():
                    item.update({
                        "message": "Folder already exists"
                    })
                else:
                    full_path.mkdir(parents=True, exist_ok=True)
            else:
                item.update({"message": "Path is required to create a folder"})
        elif action == "delete_file":
            path = item.get("path")
            if path:
                full_path = BASE_PATH / path
                if full_path.exists():
                    if full_path.is_dir():
                        shutil.rmtree(full_path)
                    else:
                        full_path.unlink()
                else:
                    item.update({"message": f"Item at {path} doesn't exist"})
            else:
                item.update({"message": "Path is required for deletion"})
        elif action == "rename_file":
            old_path = item.get("old_path")
            new_path = item.get("new_path")
            if old_path and new_path:
                full_old_path = BASE_PATH / old_path
                full_new_path = BASE_PATH / new_path
                if full_old_path.exists():
                    full_old_path.rename(full_new_path)
                else:
                    item.update({"message": f"Old path {old_path} doesn't exist"})
            else:
                item.update({"message": "Both old_path and new_path are required for renaming"})
        elif action == "list_files":
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
        elif action == "modify_content":
            path = item.get("path")
            content = item.get("content")
            if path and content:
                full_path = BASE_PATH / path
                if full_path.exists():
                    with open(full_path, "w") as f:
                        f.write(content)
                else:
                    item.update({"message": f"Item at {path} doesn't exist"})
            else:
                item.update({"message": "Path and content are required for modifying content"})
        elif action == "answer":
            pass

    return {
        "success": True,
        "message": "Chat",
        "data": response
    }