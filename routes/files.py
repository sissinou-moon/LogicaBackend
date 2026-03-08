from sre_constants import SUCCESS
from genericpath import exists
from fastapi import APIRouter, HTTPException
from pathlib import Path
import shutil

router = APIRouter()

BASE_PATH = Path("workspace")

def scanFolders(path: Path):
    items = []
    for entry in path.iterdir():
        if entry.is_dir():
            items.append({
                "name": entry.name,
                "type": "folder",
                "route": entry,
                "base": BASE_PATH,
                "children": scanFolders(entry)
            })

        else:
            items.append({
                "name": entry.name,
                "type": "file",
                "route": entry,
                "base": BASE_PATH
            })

    return items

@router.post("/create/folder")
async def function(body: dict):
    path = body.get("path")

    if not path:
        raise HTTPException(
            status_code= 420,
            detail= "Path is required"
        )

    full_path = BASE_PATH / path
    if full_path.exists():
        raise HTTPException(
            status_code= 420,
            detail= "Folder already exists"
        )

    full_path.mkdir(parents=True, exist_ok=True)

    return {
        "success": True,
        "message": "Folder created successfully",
        "data": {
            "path": path
        }
    }

@router.post("/create/file")
async def function(body: dict):
    path = body.get("path")

    if not path:
        raise HTTPException(
            status_code= 420,
            detail= "path already exisit"
        )

    full_path = BASE_PATH / path

    if full_path.exists():
        raise HTTPException(
            status_code= 420,
            detail= "File already exists"
        )

    full_path.parent.mkdir(parents=True, exist_ok=True)

    full_path.touch(exist_ok= True)

    return {
        "success": True,
        "message": "File created successfully",
        "data": {
            "path": path
        }
    }

@router.get("/list")
async def function():
    if not BASE_PATH.exists():
        BASE_PATH.mkdir(parents=True, exist_ok=True),

    return {
        "success": True,
        "message": "Here is the list",
        "data": scanFolders(BASE_PATH)
    }

@router.post("/rename")
async def function(body: dict):
    old_path = body.get("old_path")
    new_path = body.get("new_path")

    if not old_path or not new_path:
        raise HTTPException(status_code=420, detail="Old path and new path are required")

    full_old_path = BASE_PATH / old_path
    full_new_path = BASE_PATH / new_path

    if not full_old_path.exists():
        raise HTTPException(status_code=420, detail="Old path doesn't exisit")

    if full_new_path.exists():
        raise HTTPException(status_code= 420, detail= "New path already exisit")


    full_old_path.rename(full_new_path)

    return {
        "success": True,
        "message": "Folder/File was renamed successfully",
        "data": {
            "old_path": old_path,
            "new_path": new_path,
        }
    }

# ------------------ DELETE ------------------
@router.post("/delete")
async def delete_item(body: dict):
    path = body.get("path")
    if not path:
        raise HTTPException(status_code=420, detail="Path is required")

    full_path = BASE_PATH / path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Item does not exist")

    # Delete folder or file
    if full_path.is_dir():
        shutil.rmtree(full_path)
    else:
        full_path.unlink()

    return {
        "success": True,
        "message": "Item deleted successfully",
        "data": {"path": str(full_path)}
    }