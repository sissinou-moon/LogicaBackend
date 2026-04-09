from fastapi import FastAPI , WebSocket
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from middleware.errorsMiddleWare import globalErrorHandler
from routes.auth import router as auth_router
from routes.files import router as files_router
from routes.chat import router as chat_router
from memory.store import startup_maintenance
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import whisper
import time


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: prune old episodic memory
    await startup_maintenance()
    yield
    # Shutdown: nothing needed


app = FastAPI(lifespan=lifespan)

model = whisper.load_model("base.en")

SAMPLE_RATE = 16000
BUFFER = np.array([], dtype=np.float32)

CHUNK_DURATION = 3  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

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


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    global BUFFER

    await websocket.accept()
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)

            # add to buffer
            BUFFER = np.concatenate([BUFFER, chunk])

            # if we reached 3 seconds → process
            if len(BUFFER) >= CHUNK_SIZE:
                audio = BUFFER[:CHUNK_SIZE]

                # remove processed part
                BUFFER = BUFFER[CHUNK_SIZE:]

                print("🎤 Processing chunk...")

                result = model.transcribe(audio, fp16=False)
                text = result["text"].strip()

                if text:
                    print("📝 FINAL:", text)
                    await websocket.send_text(text)

    except Exception as e:
        print("WS error:", e)