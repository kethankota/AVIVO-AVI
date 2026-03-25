# AVIVO AVI (Avivo Virtual Intelligence) — Hybrid GenAI Telegram Bot

> A Hybrid Telegram bot that combines **Mini-RAG** (document Q&A) and **Vision Captioning** in a single lightweight service, backed by locally-hosted LLMs via Ollama.

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [System Design Diagram](#system-design-diagram)
4. [Models & APIs Used](#models--apis-used)
5. [Project Structure](#project-structure)
6. [Quick Start — Local (Python)](#quick-start--local-python)
7. [Quick Start — Docker Compose](#quick-start--docker-compose)
8. [Environment Variables](#environment-variables)
9. [Bot Commands](#bot-commands)
10. [Adding Knowledge Documents](#adding-knowledge-documents)
11. [Caching & Performance](#caching--performance)
12. [Design Decisions](#design-decisions)

---

## Features

| Feature | Details |
|---|---|
| **RAG Q&A** | `/ask` queries a local SQLite vector store built from `.md`/`.txt` docs |
| **Vision Captioning** | `/image` generates a two-sentence caption + 3 keyword tags using LLaVA |
| **Conversation Memory** | Keeps the last 3 turns per user for context-aware answers |
| **In-Memory Cache** | SHA-256 keyed text cache and MD5 keyed image cache prevent redundant LLM calls |
| **Source Attribution** | RAG answers include the source document excerpt |
| **Summarise** | `/summarize` condenses recent interactions with the LLM |
| **Fully Dockerised** | Single `docker-compose up` starts the bot |

---

## Architecture Overview

```
User (Telegram)
     │
     ▼
python-telegram-bot (app.py)
     │
     ├──► /ask  ──► bot/router.py ──► Cache check
     │                                    │ miss
     │                              rag/retriever.py  ← SQLite (embeddings.db)
     │                                    │
     │                              rag/generator.py  ──► Ollama (mistral:7b)
     │                                    │
     │                              utils/history.py (per-user deque)
     │
     └──► /image ──► bot/router.py ──► Cache check
                                          │ miss
                                    vision/preprocessor.py (resize → JPEG → base64)
                                          │
                                    vision/captioner.py ──► Ollama (llava:7b)
                                          │
                                    Pydantic validation → {caption, tags}
```

---

## System Design Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AVIVO AVI System                             │
│                                                                      │
│  ┌─────────────┐     ┌──────────────────────────────────────────┐   │
│  │  Telegram   │────▶│              app.py (Entry)              │   │
│  │  User       │◀────│  CommandHandler + ConversationHandler    │   │
│  └─────────────┘     └──────────────────┬───────────────────────┘   │
│                                         │                            │
│                          ┌──────────────▼────────────────┐          │
│                          │        bot/handlers.py         │          │
│                          │  handle_ask / handle_image /   │          │
│                          │  handle_summarize / handle_help│          │
│                          └──────────────┬────────────────┘          │
│                                         │                            │
│                          ┌──────────────▼────────────────┐          │
│                          │        bot/router.py           │          │
│                          │   Route: text | image | sum    │          │
│                          └──────┬────────────────┬────────┘          │
│                                 │                │                   │
│              ┌──────────────────▼──┐    ┌────────▼──────────────┐   │
│              │     RAG Pipeline    │    │   Vision Pipeline     │   │
│              │                     │    │                       │   │
│              │ rag/embedder.py     │    │ vision/preprocessor   │   │
│              │ (all-MiniLM-L6-v2  │    │ (resize → JPEG →      │   │
│              │  or arctic-embed-s) │    │  base64)              │   │
│              │         │           │    │         │             │   │
│              │ rag/retriever.py    │    │ vision/captioner.py   │   │
│              │ (cosine sim on      │    │ (LLaVA:7b via Ollama) │   │
│              │  SQLite BLOB store) │    │         │             │   │
│              │         │           │    │ Pydantic validation   │   │
│              │ rag/generator.py    │    │ {caption, tags}       │   │
│              │ (mistral:7b prompt) │    └───────────────────────┘   │
│              └─────────────────────┘                                 │
│                                                                      │
│              ┌──────────────────────────────────────────────────┐   │
│              │              Shared Utilities                     │   │
│              │  utils/cache.py   — SHA-256 / MD5 in-mem cache   │   │
│              │  utils/history.py — per-user deque (maxlen=3)    │   │
│              └──────────────────────────────────────────────────┘   │
│                                                                      │
│              ┌──────────────────────────────────────────────────┐   │
│              │             Ollama LLM Server (external)          │   │
│              │   mistral:7b  — RAG generation & summarisation   │   │
│              │   llava:7b    — Vision captioning (multimodal)   │   │
│              └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Models & APIs Used

### Embedding Model

| Model | Source | Why |
|---|---|---|
| `all-MiniLM-L6-v2` (default) | HuggingFace / sentence-transformers | Ultra-lightweight (80 MB), fast CPU inference, excellent semantic quality for English |
| `Snowflake/snowflake-arctic-embed-s` (configurable) | HuggingFace | Higher recall for technical / domain text; still small enough for CPU |

Both models produce **L2-normalised embeddings** stored as raw float32 BLOBs in SQLite. Cosine similarity is then a plain matrix dot-product — no external vector DB required.

### Generation / Vision Models (Ollama)

| Model | Task | Why |
|---|---|---|
| `mistral:7b` | RAG answer generation, history summarisation | Strong instruction-following at 7B params; runs on consumer GPU or Apple Silicon MPS |
| `llava:7b` | Image captioning + tag generation | Best open-source vision-language model at this size; structured JSON output via `"format": "json"` |

Both models are served by a shared **Ollama** instance (HTTP API on port 11434). The bot never calls the OpenAI API — all inference is local. This ensures data privacy while achieving superior performance within the limited hardware.

### Bot Framework

| Library | Version | Role |
|---|---|---|
| `python-telegram-bot` | 21.5 | Async Telegram bot, ConversationHandler for multi-step `/image` flow |
| `sentence-transformers` | 3.0.1 | Embedding model loading and inference |
| `Pillow` | 10.4.0 | Image preprocessing (resize, RGBA→RGB, JPEG encode) |
| `pydantic` | 2.12.5 | Strict validation of LLaVA JSON output |
| `requests` | 2.32.3 | Ollama HTTP calls |
| `python-dotenv` | 1.0.1 | `.env` config loading |

---

## Project Structure

```
Avivo_docker_final/
├── README.md               # Read me document for the project
├── app.py                  # Bot entry point; registers all handlers
├── requirements.txt        # Pinned Python dependencies
├── Dockerfile              # Multi-stage slim image
├── docker-compose.yml      # Compose service for the bot
├── .env                    # Secrets & config (never commit to VCS)
│
├── bot/
│   ├── handlers.py         # Telegram command handlers (ask, image, help, summarize)
│   └── router.py           # Dispatches to RAG or vision; applies cache
│
├── bot/                    # Contain the RAG and Vision routers and handler
│
├── miscellaneous/          # Contain system design diagram, sample input and  tested output images
│
├── rag/
│   ├── ingest.py           # One-time doc chunking + embedding → SQLite
│   ├── embedder.py         # SentenceTransformer wrapper (lazy singleton)
│   ├── retriever.py        # Cosine-similarity top-k retrieval from SQLite
│   └── generator.py        # Prompt builder + Ollama generate call
│
├── vision/
│   ├── preprocessor.py     # Resize, convert, base64-encode image
│   └── captioner.py        # LLaVA call + Pydantic response validation
│
├── utils/
│   ├── cache.py            # In-memory text (SHA-256) + image (MD5) cache
│   └── history.py          # Per-user sliding-window conversation history
│
├── docs/                   # Knowledge base — add your .md / .txt files here
│   ├── Apache_License_2.0.txt
│   ├── GNU_General_Public_License_(GPL)_V3.txt
│   └── Mozilla_Public_License_Version_2.0_MPL.txt
│
└── data/
    └── embeddings.db       # Auto-generated by ingest.py (SQLite)
```

---

## Quick Start — Docker Compose

### Prerequisites

- Docker Desktop (or Docker Engine + Compose v2)
- Ollama running and accessible (can be on the host machine or a remote server). Ensure the ollama url and port mentioned is .env file is pointed correctly.
- Models pulled on the Ollama host: `ollama pull mistral:7b && ollama pull llava:7b`
- A Telegram bot token from [@BotFather](https://t.me/BotFather) - is needed to have a working bot. Modify the .env values as per need. 
- for the documents provided as part of the artifacts, refer to `data` folder. sample question for these documents provided in [Bot Commands]
### Steps

```bash
# 1. clone the repository
git clone git@github.com:kethankota/AVIVO-AVI.git

# modify the env file to point to the ollama server and adjust the configurations as well
# change the current directory
cd AVIVO-AVI

# run the command to spin up the application
docker compose up -d --build

# to Stop
docker compose down -v --rmi local
```

> **Note on Ollama host networking:** If Ollama is on your local machine and the bot runs in Docker, use `host.docker.internal` (macOS/Windows) or your LAN IP (Linux) instead of `localhost` for `OLLAMA_HOST`.

### What happens on startup

1. The Docker image builds and pre-downloads the embedding model (`all-MiniLM-L6-v2`) at build time so the container is self-contained.
2. On container start, `rag/ingest.py` runs automatically to (re)build the SQLite vector store from the `docs/` folder.
3. `app.py` starts the Telegram polling loop.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_TOKEN` | *(required)* | Bot token from @BotFather |
| `OLLAMA_HOST` | `http://localhost:11434` | URL of your Ollama instance |
| `LLM_MODEL` | `mistral:7b` | Ollama model for RAG generation |
| `VISION_MODEL` | `llava:7b` | Ollama model for image captioning |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model name |
| `TOP_K` | `3` | Number of chunks to retrieve per RAG query |
| `MAX_HISTORY` | `3` | Number of conversation turns to keep per user |
| `MAX_CONTEXT_CHARS` | `3000` | Max characters of retrieved context fed to LLM |
| `CHUNK_SIZE` | `250` | Words per document chunk during ingestion |
| `CHUNK_OVERLAP` | `50` | Overlapping words between consecutive chunks |

---

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Greeting message |
| `/help` | Show command reference |
| `/ask <query>` | Ask a question answered from the knowledge base (RAG) |
| `/image` | Bot prompts you to upload a photo; returns caption + tags |
| `/summarize` | Summarise your last 3 interactions |

**Example session:**

```
You:  /ask What does the Apache License allow?
Bot:  The Apache License 2.0 allows you to freely use, modify, and distribute
      the software, including for commercial purposes, provided you include
      attribution and a copy of the license.
      Source: Apache_License_2.0.txt — "...grant you a perpetual, worldwide..."

You:  /image
Bot:  📸 Please upload the image you'd like me to summarize.
You:  [uploads photo of a lake]
Bot:  Caption: A serene scene of a pathway leading to a lake with trees and clear skies in the background.
      Tags: nature, outdoors, lake
```

---

## Adding Knowledge Documents

1. Drop any `.md` or `.txt` files into the `docs/` directory.
2. Re-run the setup
3. The old `embeddings.db` is dropped and rebuilt from scratch.

> Tip: Keep documents focused. A 3–10 page document per topic gives the retriever clean, high-quality chunks. Avoid mixing unrelated content in a single file.

---

## Caching & Performance

The bot uses a two-tier **in-memory cache** (lost on restart — by design for simplicity):

- **Text cache** (`utils/cache.py`): Keyed by `SHA-256(query.strip().lower())`. Identical questions from any user return instantly without hitting Ollama.
- **Image cache**: Keyed by `MD5(image_bytes)`. The same image sent twice is captioned only once.

This keeps **Ollama calls to a minimum** without requiring Redis or a persistent cache layer, which satisfies the assignment's "small model footprint / sensible caching" criterion.

For production, swap the dict stores in `cache.py` for a Redis client — the interface (`get_cached` / `set_cached`) is already abstracted.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **SQLite instead of a vector DB** | Zero extra services; cosine similarity via NumPy dot-product on normalised embeddings is fast enough for small-to-medium knowledge bases (< 50k chunks) |
| **Pydantic for LLaVA output** | LLaVA's JSON can be malformed; Pydantic validates and normalises `caption` and `tags` before they reach the user |
| **ConversationHandler for `/image`** | Provides a clean two-step flow: user calls `/image`, then sends the photo — no caption-hacking needed |
| **Lazy embedding singleton** | `get_model()` in `embedder.py` loads `SentenceTransformer` once per process, not per request |
| **asyncio.to_thread for CPU work** | Embedding and captioning are synchronous and CPU-bound; `to_thread` prevents blocking the async Telegram event loop |
| **`mistral:7b` over larger models** | 7B parameters fit comfortably in 8 GB VRAM (or RAM for CPU inference), giving a good accuracy-vs-speed trade-off for RAG |
