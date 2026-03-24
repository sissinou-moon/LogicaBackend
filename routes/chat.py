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


    return {
        "success": True,
        "message": "Chat",
        "data": response
    }