My AI Agent with Tiered Memory

Our AI agent runs local speech transcription using faster-whisper with GPU acceleration when available.
It supports file-based transcription through a FastAPI endpoint and returns timestamped text segments.
It also handles live microphone streaming over WebSocket with chunking and silence filtering.
The backend is optimized for fast startup and reliable fallback between CUDA and CPU modes.
It is now transcription-only (no speaker diarization or Hugging Face token dependency).

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | FastAPI (Python) |
| **Core LLMs** | DeepSeek-R1 (Local) |
| **STT / Audio** | Groq (Whisper Large V3 Turbo) local |
| **Vector DB** | ChromaDB (Cloud & Local) |
| **Relational DB** | SQLite (Local Episodic Logs) |

---

## 🧠 Memory Architecture (The Upgrade)

The system evolved from naive text retrieval to a structured **Three-Tier Memory** architecture:

1.  **Semantic Memory (ChromaDB):** Long-term storage of "facts" and "file knowledge." Uses auto-summarization to store insights rather than raw, noisy text.
2.  **Episodic Memory (SQLite + ChromaDB):** A chronological log of events (e.g., "Created folder X at 2 PM"). Allows the agent to answer questions about *past actions*.
3.  **Working Memory (Frontend State):** The immediate conversation context and current user intent.



---

## ⚙️ System Workflow

### 1. Intent Detection & Routing
Before processing a query, the system identifies the user's goal (Create, Modify, Search, or Question). This prevents unnecessary database calls and optimizes the prompt.

### 2. Context Building
Instead of dumping raw file content, the **Context Builder** constructs a structured prompt:
* **Knowledge:** High-importance snippets from Semantic Memory.
* **Recent Actions:** Last 5 events from Episodic Memory.
* **Filesystem State:** Current directory structure.

### 3. Reflection Engine (The Learning Loop)
After every successful action, the **Reflection Engine** runs in the background:
* **Summarizes** new content created.
* **Logs** the event in the episodic history.
* **Reinforces** the "importance" score of used memories, ensuring the most relevant info stays accessible.

---

## 📂 Project Structure

```bash
FinalProject/
├── memory/                  # 🧠 Memory Management Package
│   ├── models.py            # Pydantic schemas for Semantic/Episodic entries
│   └── store.py             # ChromaDB logic (Reinforcement & Decay)
├── services/                # ⚙️ Logic Layer
│   ├── aiServices.py        # LLM Orchestration
│   ├── intent.py            # Intent Classification
│   ├── summarizer.py        # Content Abstraction
│   ├── context_builder.py   # Prompt Engineering
│   └── reflection.py        # Post-action learning loop
├── routes/                  # 🌐 API Endpoints
│   ├── chat.py              # WebSocket & Chat Logic
│   └── files.py             # CRUD Operations for Filesystem
└── workspace/               # 📂 Managed User Files
```

---

## 🚀 Key Features

### 🎙️ Live Transcription & Real-time Interaction
Integrated WebSockets allow for real-time communication. Using **Deepgram Nova-3** and **Groq**, the agent supports high-speed speech-to-text, enabling hands-free file management.

### 🤖 Smart JSON Action Routing
The agent communicates via a strict JSON protocol, allowing it to chain multiple actions in a single response:
```json
[
  {
    "action": "create_folder",
    "path": "project_alpha"
  },
  {
    "action": "create_file",
    "path": "project_alpha/readme.md",
    "content": "# New Project",
    "message": "I've initialized the project directory for you."
  }
]
```

### 🎛️ Hybrid LLM Support
* **Cloud:** Gemini & HuggingFace (MiMo-V2) for high-speed, complex reasoning.
* **Local:** DeepSeek-R1 via Ollama for privacy-focused "thinking" tasks, utilizing specialized `keep_alive` and `timeout` configurations for GPU optimization.

---

## 📈 Future Roadmap
- [ ] **Memory Decay:** Implement "forgetting" logic for low-importance episodic data.
- [ ] **Multi-Agent Collaboration:** Dedicated agents for coding vs. file organization.
- [ ] **Visual Reasoning:** Using Gemini Pro Vision to analyze screenshots of the file explorer.

---
*Developed as an experiment in Agentic RAG and Autonomous File Management.*
