from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

async def globalErrorHandler(request: Request, call_next):
    try:
        return await call_next(request)
    except RequestValidationError as e:
        return JSONResponse(
            status_code=422,
            content={"success": False, "type": "validation_error", "message": str(e), "data": None}
        )

    except StarletteHTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "type": "http_error", "message": e.detail, "data": None}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "type": "server_error", "message": str(e), "data": None}
        )