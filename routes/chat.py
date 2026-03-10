from fastapi import APIRouter, HTTPException
from services.aiServices import askAI
from pathlib import Path

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

    if response.get("action") == "create_file":
        full_path = BASE_PATH / response.get("path")
        if full_path.exists():
            response.update({
                "action": "answer",
                "message": "File already exists"
            })
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.touch(exist_ok=True)
        if response.get("content") != "" or response.get("content") != None:
            with open(full_path, "w") as f:
                f.write(response.get("content"))
    elif response.get("action") == "delete_file":
        pass
    elif response.get("action") == "rename_file":
        pass
    elif response.get("action") == "list_files":
        pass
    elif response.get("action") == "answer":
        pass

    return {
        "success": True,
        "message": "Chat",
        "data": response
    }