from fastapi import FastAPI
from contextlib import asynccontextmanager
from middleware.errorsMiddleWare import globalErrorHandler
from routes.auth import router as auth_router
from routes.files import router as files_router
from routes.chat import router as chat_router
from memory.store import startup_maintenance
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: prune old episodic memory
    await startup_maintenance()
    yield
    # Shutdown: nothing needed


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(files_router, prefix="/api/v1/files")
app.include_router(chat_router, prefix="/api/v2/chat")