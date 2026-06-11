"""
vision_server — OpenAI-compatible FastVLM vision server.

Exposes:
  GET  /health                      -> {"status":"ok","model":..., "device":...}
  POST /v1/chat/completions         -> OpenAI chat-completions schema (vision)

It loads a small vision-language model (FastVLM-0.5B by default, Moondream as a
documented alternative) on CUDA, falls back to CPU with a warning (slow but works
for smoke tests). The model is loaded lazily on first request so that /health and
process startup work even without weights downloaded — this lets you smoke-test the
HTTP layer on a CPU-only box before deploying to GPU.

Env:
  LOCAL_VISION_MODEL   model id (default apple/FastVLM-0.5B)
  VISION_DEVICE        cuda | cpu | auto (default auto)
  VISION_LAZY_LOAD     1 (default) load model on first request; 0 = eager at startup
  VISION_PORT          default 8100
  HF_TOKEN             HuggingFace token, if the model is gated

Run:
  pip install -r requirements.txt
  uvicorn server:app --host 0.0.0.0 --port 8100
"""
import os
import io
import re
import json
import base64
import logging
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
log = logging.getLogger("vision-server")

# Moondream2 is the default — it loads cleanly via transformers + trust_remote_code
# and exposes a high-level .query()/.caption() API. apple/FastVLM-0.5B is NOT a
# standard transformers checkpoint (needs apple/ml-fastvlm repo code), so it is
# documented but not the default.
MODEL_ID = os.environ.get("LOCAL_VISION_MODEL", "vikhyatk/moondream2")
MODEL_REVISION = os.environ.get("LOCAL_VISION_REVISION", "").strip() or None
DEVICE_PREF = os.environ.get("VISION_DEVICE", "auto").strip().lower()
LAZY_LOAD = os.environ.get("VISION_LAZY_LOAD", "1").strip() not in ("0", "false", "no")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

app = FastAPI(title="yolo-llm-public vision server", version="1.0.0")

# Lazy globals — populated by _ensure_model().
_state: dict[str, Any] = {"model": None, "tokenizer": None, "device": None, "loaded": False}


def _resolve_device() -> str:
    if DEVICE_PREF in ("cuda", "cpu"):
        if DEVICE_PREF == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    log.warning("VISION_DEVICE=cuda but CUDA unavailable; falling back to CPU.")
                    return "cpu"
            except Exception:
                return "cpu"
        return DEVICE_PREF
    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _ensure_model() -> None:
    """Load the model+tokenizer once. Raises on failure (caller maps to HTTP 503)."""
    if _state["loaded"]:
        return
    import torch  # noqa: F401  (import here so /health works without torch installed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device()
    if device == "cpu":
        log.warning("Loading vision model on CPU — this is SLOW. Use a GPU for real workloads.")
    log.info(f"Loading vision model {MODEL_ID} on {device} ...")

    import torch as _torch
    dtype = _torch.float16 if device == "cuda" else _torch.float32

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=HF_TOKEN, revision=MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        token=HF_TOKEN,
        revision=MODEL_REVISION,
    ).to(device)
    model.eval()

    _state.update(model=model, tokenizer=tok, device=device, loaded=True)
    log.info(f"Vision model loaded on {device}")


if not LAZY_LOAD:
    try:
        _ensure_model()
    except Exception as e:  # pragma: no cover
        log.error(f"Eager model load failed: {e!r}")


# ── OpenAI-compatible schema (subset) ──────────────────────────────────
class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float | None = 0.0
    max_tokens: int | None = 64
    response_format: dict | None = None


def _load_image_from_content(messages: list[dict]):
    """Extract the (text_prompt, PIL.Image) from OpenAI-style multimodal messages."""
    from PIL import Image
    import httpx

    text_prompt = ""
    image = None
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            text_prompt = text_prompt or content
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                text_prompt = text_prompt or part.get("text", "")
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    b64 = url.split(",", 1)[1]
                    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                elif url:
                    r = httpx.get(url, timeout=15.0, follow_redirects=True,
                                  headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()
                    image = Image.open(io.BytesIO(r.content)).convert("RGB")
    return text_prompt, image


def _run_inference(text_prompt: str, image, max_tokens: int) -> str:
    """Run the loaded VLM. Returns raw model text output.

    Uses transformers generate with the model's chat/processor when available.
    FastVLM and Moondream both expose trust_remote_code generate paths; we use a
    generic best-effort call and let _coerce_json clean up the result.
    """
    import torch
    model = _state["model"]
    tok = _state["tokenizer"]
    device = _state["device"]

    # Moondream2 (new API ≥2025): model.query(image, question) -> {"answer": ...}
    if hasattr(model, "query"):
        try:
            out = model.query(image, text_prompt)
            ans = out.get("answer") if isinstance(out, dict) else out
            if ans:
                return str(ans)
        except Exception as e:
            log.warning(f"moondream .query failed, trying encode_image: {e!r}")
    # Moondream2 (old API): encode_image + answer_question
    if hasattr(model, "encode_image") and hasattr(model, "answer_question"):
        enc = model.encode_image(image)
        return model.answer_question(enc, text_prompt, tok)

    # Generic transformers path (FastVLM-style): build chat prompt with <image>.
    prompt = f"<image>\n{text_prompt}"
    inputs = tok(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=False)
    try:
        # If the model accepts pixel inputs via processor, attach them.
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
        proc_inputs = proc(text=prompt, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**proc_inputs, **gen_kwargs)
        return proc.batch_decode(out_ids, skip_special_tokens=True)[0]
    except Exception:
        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
        return tok.batch_decode(out_ids, skip_special_tokens=True)[0]


def _coerce_json(raw: str) -> str:
    """Coerce model output into a {"name","symbol"} JSON string."""
    raw = (raw or "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return json.dumps({"name": obj.get("name", ""), "symbol": obj.get("symbol", "")})
        except Exception:
            pass
    return json.dumps({"name": raw[:30], "symbol": ""})


@app.get("/health")
def health():
    device = _state["device"] or _resolve_device()
    return {"status": "ok", "model": MODEL_ID, "device": device, "loaded": _state["loaded"]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        _ensure_model()
    except Exception as e:
        log.error(f"model load failed: {e!r}")
        return JSONResponse(status_code=503, content={"error": f"model load failed: {e!s}"})

    try:
        text_prompt, image = _load_image_from_content(req.messages)
        if image is None:
            return JSONResponse(status_code=400, content={"error": "no image in messages"})
        raw = _run_inference(text_prompt, image, req.max_tokens or 64)
        content = _coerce_json(raw)
    except Exception as e:
        log.error(f"inference err: {e!r}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("VISION_PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port)
