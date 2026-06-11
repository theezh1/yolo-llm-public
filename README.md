# yolo-llm-public

Telegram bot that takes a meme-token image (photo or tweet image URL), runs **YOLO11n**
to detect and square-crop the main object, and uses a **vision LLM** to extract the
token `{name, symbol}` — then looks it up on DexScreener. Vision runs via the **Groq API**
or a **local FastVLM** server, switchable with one env var.

## Architecture

```
Telegram ──photo / image URL──► bot.py
                                  │
                  ┌───────────────┼────────────────┐
                  ▼               ▼                 ▼
              YOLO11n      vision extract      DexScreener
            (crop main    {name,symbol}        (lookup name)
             object)            │
                               VISION_BACKEND
                          ┌──────┴───────┐
                          ▼              ▼
                     Groq API      vision_server/  (FastVLM, :8100,
                    (cloud)         OpenAI-compatible /v1/chat/completions)
```

## Quick start (RunPod)

1. New Pod → **PyTorch** preset (template "RunPod PyTorch 2.x"), GPU, region **US-East**.
2. In the pod terminal:
   ```bash
   git clone https://github.com/theezh1/yolo-llm-public.git
   cd yolo-llm-public
   cp .env.example .env
   nano .env                     # set BOT_TOKEN (+ GROQ_API_KEY for groq mode)
   docker compose up --build     # bot + vision-server
   ```
   Or without Docker (PyTorch already installed in the preset):
   ```bash
   pip install -r requirements.txt
   # groq mode — just run the bot:
   VISION_BACKEND=groq python bot.py
   # local mode — start the vision server first, then the bot:
   pip install -r vision_server/requirements.txt
   uvicorn server:app --host 0.0.0.0 --port 8100 --app-dir vision_server &
   VISION_BACKEND=local python bot.py
   ```
3. DM the bot or add it to a group. Send a photo, `/ai <text>` + photo, or paste an
   image URL (e.g. `https://pbs.twimg.com/media/...`).

> `ALLOWED_CHAT_ID=0` (default) serves **all** chats — fine for testing. Set it to your
> group id to lock the bot down.

## VISION_BACKEND

| `VISION_BACKEND` | Vision provider | Needs | Latency | Notes |
|---|---|---|---|---|
| `groq` (default) | Groq Llama 4 Scout (cloud) | `GROQ_API_KEY` | ~300–800 ms | no GPU needed for vision |
| `local` | FastVLM via `vision_server/` | GPU + model download | depends on GPU | private, no API key |

Switching is just `VISION_BACKEND=groq` ↔ `VISION_BACKEND=local`; the bot uses the same
`{name,symbol}` contract either way.

## Environment

| Var | Default | Meaning |
|---|---|---|
| `BOT_TOKEN` | — (required) | Telegram bot token |
| `ALLOWED_CHAT_ID` | `0` | `0` = all chats; else lock to this chat id |
| `VISION_BACKEND` | `groq` | `groq` or `local` |
| `GROQ_API_KEY` | — | required for `groq` mode |
| `GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq vision model |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq text model (for `/ai <text>`) |
| `LOCAL_VISION_URL` | `http://localhost:8100/v1/chat/completions` | local vision endpoint |
| `LOCAL_VISION_MODEL` | `apple/FastVLM-0.5B` | local VLM id |
| `YOLO_MODEL` | `yolo11n.pt` | auto-downloaded on first run |

## GPU requirements (local mode)

| Model | VRAM | Notes |
|---|---|---|
| `apple/FastVLM-0.5B` | ~2 GB | default; fits any 3090 / 4090 / A40 / L4 |
| `vikhyatk/moondream2` | ~4 GB | alternative if FastVLM is gated / hard to load |

CPU fallback works (set `VISION_DEVICE=cpu`) but is slow — for smoke tests only.

> **FastVLM note:** if `apple/FastVLM-0.5B` requires gated HuggingFace access or its
> `trust_remote_code` path fails on your transformers version, set
> `LOCAL_VISION_MODEL=vikhyatk/moondream2` (Moondream) and add its deps — both speak the
> same OpenAI-compatible endpoint, no bot changes needed. Provide `HF_TOKEN` for gated repos.

## Testing

| Test | Where | Note |
|---|---|---|
| Bot import / handlers | any host | `python -c "import bot"` (needs `BOT_TOKEN` set) |
| Vision server HTTP layer | any host (CPU) | `uvicorn ... &` then `curl localhost:8100/health` → 200 |
| **Full vision inference** | **GPU host** | loads FastVLM/Moondream; the model is lazy-loaded on first `/v1/chat/completions` |

The vision server lazy-loads weights on the first inference request, so `/health` and
process startup succeed on a CPU-only box without any model download — the real vision
test requires a GPU server.

## Files

| Path | What |
|---|---|
| `bot.py` | Telegram bot, YOLO crop, DexScreener, vision dispatch |
| `local_vision.py` | client for the local vision server |
| `vision_server/server.py` | FastVLM OpenAI-compatible server (:8100) |
| `Dockerfile` / `docker-compose.yml` | container build + 2-service stack |
| `.env.example` | all config knobs |
