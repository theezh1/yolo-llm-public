"""
vision_server — OpenAI-compatible vision server (FastVLM / Moondream).

Exposes:
  GET  /health                      -> {"status":"ok","model":..., "device":..., "backend":...}
  POST /v1/chat/completions         -> OpenAI chat-completions schema (vision)

Two backends, selected by the model id in LOCAL_VISION_MODEL:

  * FastVLM  — if the id contains "fastvlm" (case-insensitive). Apple's
    apple/FastVLM-0.5B is a LLaVA-architecture research model with a FastViT-HD
    encoder. It is NOT a standard transformers checkpoint; it needs the apple/ml-fastvlm
    code (a fork of LLaVA) on the import path. Loaded via
    llava.model.builder.load_pretrained_model. Target latency ~15-50ms on GPU.

  * Moondream — fallback for any other id (e.g. vikhyatk/moondream2). Loads cleanly
    via transformers trust_remote_code and exposes .query()/.encode_image(). ~415ms.

For FastVLM, LOCAL_VISION_MODEL should point at the unpacked checkpoint directory on
disk (e.g. /root/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3) OR contain the
word "fastvlm" while FASTVLM_MODEL_PATH gives the on-disk path. Either works.

The model is loaded lazily on first request so /health and process startup work even
without weights / heavy deps installed — lets you smoke-test the HTTP layer on a
CPU-only box before deploying to GPU.

Env:
  LOCAL_VISION_MODEL   model id or checkpoint path (default vikhyatk/moondream2)
  FASTVLM_MODEL_PATH   on-disk checkpoint dir for FastVLM (overrides LOCAL_VISION_MODEL path)
  FASTVLM_CONV_MODE    LLaVA conversation template (default qwen_2 — correct for 0.5B/1.5B/7B)
  VISION_DEVICE        cuda | cpu | auto (default auto)
  VISION_LAZY_LOAD     1 (default) load model on first request; 0 = eager at startup
  VISION_PORT          default 8100
  HF_TOKEN             HuggingFace token, if a model is gated

Run:
  pip install -r requirements.txt          # moondream path
  # FastVLM path: install apple/ml-fastvlm into its own venv (transformers==4.48.3)
  uvicorn server:app --host 0.0.0.0 --port 8100
"""
import os
import io
import re
import json
import time
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

MODEL_ID = os.environ.get("LOCAL_VISION_MODEL", "vikhyatk/moondream2")
MODEL_REVISION = os.environ.get("LOCAL_VISION_REVISION", "").strip() or None
DEVICE_PREF = os.environ.get("VISION_DEVICE", "auto").strip().lower()
LAZY_LOAD = os.environ.get("VISION_LAZY_LOAD", "1").strip() not in ("0", "false", "no")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
FASTVLM_MODEL_PATH = os.environ.get("FASTVLM_MODEL_PATH", "").strip() or None
FASTVLM_CONV_MODE = os.environ.get("FASTVLM_CONV_MODE", "qwen_2").strip()


def _is_fastvlm() -> bool:
    """FastVLM backend selected when the model id or path mentions fastvlm."""
    hay = f"{MODEL_ID} {FASTVLM_MODEL_PATH or ''}".lower()
    return "fastvlm" in hay


def _is_lfm2() -> bool:
    """LFM2 backend selected when the model id mentions lfm2 (LiquidAI LFM2-VL)."""
    return "lfm2" in MODEL_ID.lower()


def _select_backend() -> str:
    if _is_lfm2():
        return "lfm2"
    if _is_fastvlm():
        return "fastvlm"
    return "moondream"


# backend = "lfm2" | "fastvlm" | "moondream"
BACKEND = _select_backend()

app = FastAPI(title="yolo-llm-public vision server", version="2.0.0")

# Lazy globals — populated by _ensure_model().
_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "image_processor": None,
    "device": None,
    "loaded": False,
    "backend": BACKEND,
}


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
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ── Model loading ──────────────────────────────────────────────────────
def _load_fastvlm(device: str) -> None:
    """Load Apple FastVLM via the apple/ml-fastvlm (LLaVA fork) code path."""
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_path = FASTVLM_MODEL_PATH or MODEL_ID
    model_path = os.path.expanduser(model_path)
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"FastVLM checkpoint dir not found: {model_path!r}. "
            "Download it (get_models.sh) and set FASTVLM_MODEL_PATH."
        )

    model_name = get_model_name_from_path(model_path)
    log.info(f"Loading FastVLM checkpoint {model_path} (name={model_name}) on {device} ...")
    tokenizer, model, image_processor, _ctx = load_pretrained_model(
        model_path, None, model_name, device=device
    )
    model.eval()
    # Make greedy generate deterministic re: pad token.
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    _state.update(
        model=model, tokenizer=tokenizer, image_processor=image_processor,
        device=device, loaded=True, backend="fastvlm",
    )
    log.info("FastVLM loaded.")


def _load_lfm2(device: str) -> None:
    """Load LiquidAI LFM2-VL (standard transformers >= 4.57, AutoModelForImageTextToText).

    LFM2-VL is a liquid-neural-net VLM that loads as a normal transformers checkpoint
    (unlike FastVLM). ChatML template, <image> sentinel, processor.apply_chat_template.
    Works for both the base LFM2-VL-450M and the LFM2.5-VL-*-Extract variants.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    log.info(f"Loading LFM2-VL checkpoint {MODEL_ID} on {device} (dtype={dtype}) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto" if device == "cuda" else None,
        dtype=dtype,
        trust_remote_code=True,
        token=HF_TOKEN,
        revision=MODEL_REVISION,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=HF_TOKEN, revision=MODEL_REVISION
    )
    # image_processor slot reused to carry the processor for LFM2.
    _state.update(model=model, tokenizer=processor, image_processor=processor,
                  device=device, loaded=True, backend="lfm2")
    log.info("LFM2-VL loaded.")


def _load_moondream(device: str) -> None:
    """Load Moondream (or any generic trust_remote_code VLM) via transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=HF_TOKEN, revision=MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=dtype,
        token=HF_TOKEN, revision=MODEL_REVISION,
    ).to(device)
    model.eval()
    _state.update(model=model, tokenizer=tok, image_processor=None,
                  device=device, loaded=True, backend="moondream")
    log.info(f"Moondream/generic model loaded on {device}")


def _ensure_model() -> None:
    """Load the model once. Raises on failure (caller maps to HTTP 503)."""
    if _state["loaded"]:
        return
    device = _resolve_device()
    if device == "cpu":
        log.warning("Loading vision model on CPU — this is SLOW. Use a GPU for real workloads.")
    if BACKEND == "lfm2":
        _load_lfm2(device)
    elif BACKEND == "fastvlm":
        _load_fastvlm(device)
    else:
        _load_moondream(device)


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
    """Extract (text_prompt, PIL.Image) from OpenAI-style multimodal messages."""
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


# ── Inference ──────────────────────────────────────────────────────────
def _infer_fastvlm(text_prompt: str, image, max_tokens: int) -> str:
    """Run FastVLM (LLaVA fork). Builds the qwen_2 conv prompt with <image>."""
    import torch
    from llava.conversation import conv_templates
    from llava.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    )
    from llava.mm_utils import tokenizer_image_token, process_images

    model = _state["model"]
    tok = _state["tokenizer"]
    image_processor = _state["image_processor"]
    device = _state["device"]

    qs = text_prompt
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[FASTVLM_CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )
    image_tensor = process_images([image], image_processor, model.config)[0]
    images = image_tensor.unsqueeze(0).to(device)
    if device == "cuda":
        images = images.half()

    with torch.inference_mode():
        out_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=max_tokens,
            use_cache=True,
        )
    return tok.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def _infer_lfm2(text_prompt: str, image, max_tokens: int) -> str:
    """Run LiquidAI LFM2-VL via ChatML chat template.

    For the base model: user message with [image, text].
    For the *-Extract* variants: a system prompt describing the JSON fields to
    extract plus a user image — the Extract models are tuned to emit pure JSON.
    Gen params per model card: temperature=0.1, min_p=0.15, repetition_penalty=1.05.
    """
    import torch

    model = _state["model"]
    processor = _state["image_processor"]
    device = _state["device"]

    is_extract = "extract" in MODEL_ID.lower()
    if is_extract:
        # Extract variant: YAML-style field schema in system prompt, image in user
        # turn. The Extract models are tuned to emit pure JSON. transformers v5
        # requires every message["content"] to be a list of typed parts, so the
        # system text is wrapped too. The symbol is intentionally left loose — the
        # server's _coerce_json/_derive_symbol cleans it up from the name.
        system_prompt = (
            "Extract the following from the image:\n\n"
            "name: The name of the main character, subject, or prominent text shown in the image\n"
            "symbol: An uppercase ticker symbol for a meme token based on the name\n\n"
            "Respond with only a JSON object."
        )
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}]},
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    if is_extract:
        # Extract model card recommends greedy decoding for deterministic JSON.
        gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=False)
    else:
        # Base LFM2-VL sampling params per model card.
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            min_p=0.15,
            repetition_penalty=1.05,
        )
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)

    # Strip the prompt tokens; decode only the newly generated continuation.
    gen_only = out_ids[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_only, skip_special_tokens=True)
    return text.strip()


def _infer_moondream(text_prompt: str, image, max_tokens: int) -> str:
    import torch
    model = _state["model"]
    tok = _state["tokenizer"]

    if hasattr(model, "query"):
        try:
            out = model.query(image, text_prompt)
            ans = out.get("answer") if isinstance(out, dict) else out
            if ans:
                return str(ans)
        except Exception as e:
            log.warning(f"moondream .query failed, trying encode_image: {e!r}")
    if hasattr(model, "encode_image") and hasattr(model, "answer_question"):
        enc = model.encode_image(image)
        return model.answer_question(enc, text_prompt, tok)
    # last-resort generic generate
    prompt = f"<image>\n{text_prompt}"
    inputs = tok(prompt, return_tensors="pt").to(_state["device"])
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tok.batch_decode(out_ids, skip_special_tokens=True)[0]


def _run_inference(text_prompt: str, image, max_tokens: int) -> tuple[str, float]:
    """Dispatch to the active backend. Returns (raw_text, inference_ms)."""
    t0 = time.perf_counter()
    if _state["backend"] == "lfm2":
        raw = _infer_lfm2(text_prompt, image, max_tokens)
    elif _state["backend"] == "fastvlm":
        raw = _infer_fastvlm(text_prompt, image, max_tokens)
    else:
        raw = _infer_moondream(text_prompt, image, max_tokens)
    ms = (time.perf_counter() - t0) * 1000.0
    log.info(f"inference backend={_state['backend']} {ms:.1f}ms -> {raw[:80]!r}")
    return raw, ms


def _derive_symbol(name: str) -> str:
    """UPPERCASE letters-only, 3-10 chars, derived from name."""
    letters = re.sub(r"[^A-Za-z]", "", name or "").upper()
    return letters[:10] if len(letters) >= 3 else letters


def _coerce_json(raw: str) -> str:
    """Coerce model output into a {"name","symbol"} JSON string."""
    raw = (raw or "").strip()
    name, symbol = "", ""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            name = str(obj.get("name", "") or "").strip()
            symbol = str(obj.get("symbol", "") or "").strip()
        except Exception:
            pass
    if not name:
        name = raw[:30]
    if not symbol or not re.fullmatch(r"[A-Za-z0-9]{2,10}", symbol):
        symbol = _derive_symbol(name)
    return json.dumps({"name": name, "symbol": symbol})


@app.get("/health")
def health():
    device = _state["device"] or _resolve_device()
    return {
        "status": "ok",
        "model": MODEL_ID,
        "backend": _state["backend"],
        "device": device,
        "loaded": _state["loaded"],
    }


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
        raw, ms = _run_inference(text_prompt, image, req.max_tokens or 64)
        content = _coerce_json(raw)
    except Exception as e:
        log.error(f"inference err: {e!r}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": MODEL_ID,
        "backend": _state["backend"],
        "inference_ms": round(ms, 1),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("VISION_PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port)
