from fastapi import HTTPException
from fastapi import APIRouter

router = APIRouter()

@router.post("/auth")
async def auth(body: dict):

    email = body.get("email")
    password = body.get("password")

    if not email or not password:
        raise HTTPException(
            status_code=420,
            detail="Email and password are required"
        )

    return {
        "success": True,
        "message": "User logged in successfully",
        "data": {
            "email": email,
            "password": password
        }
    }