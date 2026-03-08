from fastapi import FastAPI
from middleware.errorsMiddleWare import globalErrorHandler
from routes.auth import router as auth_router
from routes.files import router as files_router

app = FastAPI()


app.include_router(auth_router, prefix="/api/v1")
app.include_router(files_router, prefix="/api/v1")