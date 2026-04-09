"""
Upgraded transcription backend:
- Live mic streaming via WebSocket (VAD + chunking)
- MP3/audio file upload + transcription
- Speaker diarization via pyannote.audio
- Faster-whisper for speed + quality
"""

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import tempfile
import os
import json
import time
import warnings
from pathlib import Path

# ── Faster-whisper (much faster than openai-whisper, same quality) ──────────
from faster_whisper import WhisperModel

# ── Speaker diarization ──────────────────────────────────────────────────────
# pip install pyannote.audio
# Requires HuggingFace token + accepting model terms at:
# https://huggingface.co/pyannote/speaker-diarization-3.1
warnings.filterwarnings(
    "ignore",
    message=r"torchcodec is not installed correctly so built-in audio decoding will fail\..*",
    category=UserWarning,
    module=r"pyannote\.audio\.core\.io",
)
from pyannote.audio import Pipeline
import torch
import soundfile as sf
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto").strip().lower()
if HF_TOKEN and not HUGGINGFACE_HUB_TOKEN:
    # Keep tooling compatible with libs that read only this alias.
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

FFMPEG_BIN = r"C:\ffmpeg\bin"
if hasattr(os, "add_dll_directory") and os.path.isdir(FFMPEG_BIN):
    os.add_dll_directory(FFMPEG_BIN)

SAMPLE_RATE = 16000
CHUNK_DURATION = 2          # seconds per live chunk (shorter = more responsive)
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
SILENCE_THRESHOLD = 0.01    # RMS below this = silence, skip processing


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    gpu_available = torch.cuda.is_available()
    if WHISPER_DEVICE == "cuda" and not gpu_available:
        print("⚠️  WHISPER_DEVICE=cuda but CUDA is unavailable; falling back to CPU.")
    use_cuda = gpu_available and WHISPER_DEVICE != "cpu"
    model_device = "cuda" if use_cuda else "cpu"
    model_compute_type = "float16" if use_cuda else "int8"

    if use_cuda:
        print(f"🚀 Using GPU for Whisper: {torch.cuda.get_device_name(0)}")
    else:
        print("🖥️  Using CPU for Whisper.")

    print("⏳ Loading Whisper model (large-v3)...")
    app.state.whisper = WhisperModel(
        "small.en",          # best quality; use "medium" or "small.en" for speed
        device=model_device,
        compute_type=model_compute_type,
    )
    print("✅ Whisper loaded")

    print("⏳ Loading diarization pipeline...")
    try:
        if not HF_TOKEN:
            print("⚠️  HF_TOKEN not found in environment; diarization may fail.")
        app.state.diarize = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        if use_cuda:
            app.state.diarize = app.state.diarize.to(torch.device("cuda"))
        print("✅ Diarization loaded")
    except Exception as e:
        print(f"⚠️  Diarization unavailable: {e}. Set HF_TOKEN env var.")
        app.state.diarize = None

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def transcribe_audio(model, audio_path: str, language: str = None):
    """Transcribe with faster-whisper, return segments."""
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,                # built-in VAD to remove silence
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
    )
    return list(segments), info


def diarize_audio(pipeline, audio_path: str):
    """Return speaker turn dict: {(start, end): speaker_label}"""
    if pipeline is None:
        return {}
    # Preload audio in-memory so pyannote does not depend on torchcodec.
    waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T)  # (channels, time)
    diarization = pipeline({"waveform": waveform, "sample_rate": int(sample_rate)})
    turns = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns[(turn.start, turn.end)] = speaker
    return turns


def assign_speaker(start: float, end: float, turns: dict) -> str:
    """Find the speaker with most overlap for this segment."""
    best_speaker = "Unknown"
    best_overlap = 0.0
    for (t_start, t_end), speaker in turns.items():
        overlap = max(0, min(end, t_end) - max(start, t_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def merge_segments(segments, turns: dict):
    """Combine whisper segments with speaker labels."""
    result = []
    for seg in segments:
        speaker = assign_speaker(seg.start, seg.end, turns)
        result.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "speaker": speaker,
            "text": seg.text.strip(),
        })
    return result


# ── File upload endpoint ─────────────────────────────────────────────────────

@app.post("/api/v2/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    """Upload any audio file, get back diarized transcript."""
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments, info = transcribe_audio(app.state.whisper, tmp_path)
        turns = diarize_audio(app.state.diarize, tmp_path)
        merged = merge_segments(segments, turns)

        return JSONResponse({
            "language": info.language,
            "duration": round(info.duration, 2),
            "segments": merged,
        })
    finally:
        os.unlink(tmp_path)


# ── Live WebSocket endpoint ──────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")

    buffer = np.array([], dtype=np.float32)

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) >= CHUNK_SIZE:
                audio = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]

                # Skip near-silence chunks
                if rms(audio) < SILENCE_THRESHOLD:
                    continue

                # Write to temp wav for whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, SAMPLE_RATE)
                    wav_path = f.name

                try:
                    segments, _ = transcribe_audio(app.state.whisper, wav_path)
                    text = " ".join(seg.text.strip() for seg in segments)
                    if text:
                        await websocket.send_text(json.dumps({
                            "type": "transcript",
                            "text": text,
                            "speaker": "Live",
                        }))
                finally:
                    os.unlink(wav_path)

    except Exception as e:
        print("WS error:", e)
    finally:
        print("🔌 Client disconnected")