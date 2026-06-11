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
# ONNX backend (onnx-community/FastVLM-0.5B-ONNX — 3-graph transformers.js layout)
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "").strip() or None
ONNX_QUANT = os.environ.get("ONNX_QUANT", "q4").strip()          # decoder quant: fp16|q4|int8|...
ONNX_ENC_QUANT = os.environ.get("ONNX_ENC_QUANT", "fp16").strip()  # vision/embed quant: fp16|<empty for fp32>
ONNX_IMAGE_TOKEN = int(os.environ.get("ONNX_IMAGE_TOKEN", "151646"))


def _is_fastvlm() -> bool:
    """FastVLM backend selected when the model id or path mentions fastvlm."""
    hay = f"{MODEL_ID} {FASTVLM_MODEL_PATH or ''}".lower()
    return "fastvlm" in hay


def _is_lfm2() -> bool:
    """LFM2 backend selected when the model id mentions lfm2 (LiquidAI LFM2-VL)."""
    return "lfm2" in MODEL_ID.lower()


def _is_onnx() -> bool:
    """ONNX backend selected when the model id or ONNX_MODEL_PATH mentions onnx."""
    hay = f"{MODEL_ID} {os.environ.get('ONNX_MODEL_PATH', '')}".lower()
    return "onnx" in hay


def _is_llama() -> bool:
    """Llama-Vision backend selected when the model id mentions llama (Llama-3.2-Vision)."""
    return "llama" in MODEL_ID.lower()


def _is_ocr() -> bool:
    """PaddleOCR backend selected when LOCAL_VISION_MODEL or VISION_BACKEND == 'ocr'."""
    return MODEL_ID.strip().lower() == "ocr" or \
        os.environ.get("VISION_BACKEND", "").strip().lower() == "ocr"


def _select_backend() -> str:
    if _is_ocr():
        return "ocr"
    if _is_onnx():
        return "onnx"
    if _is_lfm2():
        return "lfm2"
    if _is_llama():
        return "llama"
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
    "onnx_cfg": None,
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


def _load_llama(device: str) -> None:
    """Load Llama-3.2-11B-Vision-Instruct in 4-bit (bitsandbytes).

    Llama-3.2-Vision is an Mllama architecture VLM. It loads via
    MllamaForConditionalGeneration + AutoProcessor. We quantise to 4-bit with
    BitsAndBytesConfig (nf4, double-quant, bf16 compute) — fits the 11B in ~7GB.
    Works with the open unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit
    (no gating) or meta-llama/Llama-3.2-11B-Vision-Instruct (gated, needs HF_TOKEN).
    The chat template injects an <|image|> tile for the image part.
    """
    import torch
    from transformers import (
        MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    )

    # The unsloth *-bnb-4bit checkpoints ship pre-quantised; passing a fresh
    # BitsAndBytesConfig is still accepted and is required for non-prequantised
    # repos (meta-llama). bf16 compute keeps quality high on the A40.
    is_prequant = "4bit" in MODEL_ID.lower() or "bnb" in MODEL_ID.lower()
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    log.info(
        f"Loading Llama-3.2-Vision {MODEL_ID} 4-bit on {device} "
        f"(prequant={is_prequant}) ..."
    )
    kwargs: dict[str, Any] = dict(
        device_map="auto" if device == "cuda" else None,
        token=HF_TOKEN,
        revision=MODEL_REVISION,
    )
    # Prequantised repos already carry quantization_config in their config.json;
    # supplying our own would override the compute dtype, which is fine, but we
    # keep it for both paths to force nf4 + bf16 compute deterministically.
    kwargs["quantization_config"] = quant_cfg
    model = MllamaForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, token=HF_TOKEN, revision=MODEL_REVISION
    )
    _state.update(model=model, tokenizer=processor, image_processor=processor,
                  device=device, loaded=True, backend="llama")
    log.info("Llama-3.2-Vision (4-bit) loaded.")


def _load_ocr(device: str) -> None:
    """Load PaddleOCR (detector + recogniser). Deterministic text reader, no LLM.

    Tries GPU first (paddlepaddle-gpu); PaddleOCR falls back to CPU automatically
    if the GPU build/CUDA isn't available. English model, angle classifier on.
    """
    from paddleocr import PaddleOCR

    use_gpu = device == "cuda"
    log.info(f"Loading PaddleOCR (use_gpu={use_gpu}) ...")
    ocr = None
    # PaddleOCR's constructor signature changed across versions (use_gpu was
    # removed in 2.7+, replaced by the paddle device env). Try the modern call
    # first, fall back to the legacy one.
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
    except TypeError:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=use_gpu)
    _state.update(model=ocr, tokenizer=None, image_processor=None,
                  device=device, loaded=True, backend="ocr")
    log.info("PaddleOCR loaded.")


def _load_onnx(device: str) -> None:
    """Load onnx-community/FastVLM-0.5B-ONNX via raw onnxruntime (3-graph layout).

    The HF repo is a transformers.js export: three separate ONNX graphs
    (vision_encoder, embed_tokens, decoder_model_merged) that optimum's
    ORTModelForVision2Seq cannot orchestrate (llava_qwen2 is unsupported there).
    We drive them by hand. The AutoProcessor handles <image> placeholder expansion
    and pixel preprocessing; we splice the vision features into the input embeddings
    at the image-token positions, then run a greedy KV-cache decode loop.

    Quant is picked per-graph via ONNX_QUANT (decoder) / ONNX_ENC_QUANT (encoder+embed).
    """
    import onnxruntime as ort
    from transformers import AutoTokenizer, CLIPImageProcessor
    from huggingface_hub import snapshot_download

    src = ONNX_MODEL_PATH or MODEL_ID
    if os.path.isdir(os.path.expanduser(src)):
        root = os.path.expanduser(src)
    else:
        # llava_qwen2 isn't a registered transformers arch, so we pull files
        # explicitly (no AutoModel) and drive the ONNX graphs by hand.
        quants = {ONNX_QUANT, ONNX_ENC_QUANT}
        patterns = ["*.json", "*.txt", "tokenizer*", "vocab*", "merges*",
                    "special_tokens*", "added_tokens*"]
        for stem in ("vision_encoder", "embed_tokens", "decoder_model_merged"):
            for q in quants:
                patterns.append(f"onnx/{stem}_{q}.onnx")
                patterns.append(f"onnx/{stem}_{q}.onnx_data")
        root = snapshot_download(src, token=HF_TOKEN, revision=MODEL_REVISION,
                                 allow_patterns=patterns)
    onnx_dir = os.path.join(root, "onnx")

    def _f(stem: str, quant: str) -> str:
        suffix = f"_{quant}" if quant else ""
        p = os.path.join(onnx_dir, f"{stem}{suffix}.onnx")
        if not os.path.isfile(p):
            p = os.path.join(onnx_dir, f"{stem}.onnx")  # fall back to fp32
        return p

    if device == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    def _sess(path: str):
        return ort.InferenceSession(path, so, providers=providers)

    log.info(f"Loading ONNX FastVLM from {onnx_dir} "
             f"(dec={ONNX_QUANT}, enc={ONNX_ENC_QUANT}) on {device} ...")
    vis = _sess(_f("vision_encoder", ONNX_ENC_QUANT))
    emb = _sess(_f("embed_tokens", ONNX_ENC_QUANT))
    dec = _sess(_f("decoder_model_merged", ONNX_QUANT))

    # llava_qwen2 has no transformers arch, but its sub-components load directly:
    # the LM tokenizer is plain Qwen2, the image processor is CLIPImageProcessor.
    tokenizer = AutoTokenizer.from_pretrained(root, token=HF_TOKEN)
    image_processor = CLIPImageProcessor.from_pretrained(root, token=HF_TOKEN)

    import json as _json
    cfg = {}
    cfg_path = os.path.join(root, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            cfg = _json.load(fh)

    # decoder input dtype for inputs_embeds / kv cache (fp16 graph wants float16)
    dec_in = {i.name: i.type for i in dec.get_inputs()}
    emb_out_t = next(o.type for o in emb.get_outputs())
    feat_t = "float16" if "float16" in str(emb_out_t) else "float32"
    kv_t = "float16" if "float16" in dec_in.get("past_key_values.0.key", "float32") else "float32"

    n_heads = int(cfg.get("num_attention_heads", 14))
    hidden = int(cfg.get("hidden_size", 896))
    _state.update(
        model={"vis": vis, "emb": emb, "dec": dec},
        tokenizer=tokenizer, image_processor=image_processor,
        device=device, loaded=True, backend="onnx",
        onnx_cfg={
            "n_layers": int(cfg.get("num_hidden_layers", 24)),
            "n_kv_heads": int(cfg.get("num_key_value_heads", 2)),
            "head_dim": hidden // n_heads,
            "image_token": int(cfg.get("image_token_index", ONNX_IMAGE_TOKEN)),
            "feat_dtype": feat_t,
            "kv_dtype": kv_t,
        },
    )
    log.info(f"ONNX FastVLM loaded (feat={feat_t}, kv={kv_t}, img_tok={_state['onnx_cfg']['image_token']}).")


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
    if BACKEND == "ocr":
        _load_ocr(device)
    elif BACKEND == "onnx":
        _load_onnx(device)
    elif BACKEND == "lfm2":
        _load_lfm2(device)
    elif BACKEND == "llama":
        _load_llama(device)
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


def _infer_onnx(text_prompt: str, image, max_tokens: int) -> str:
    """Greedy decode onnx-community/FastVLM-0.5B-ONNX over 3 raw ONNX graphs.

    Steps:
      1. processor builds input_ids (with <image> placeholders expanded) + pixel_values.
      2. embed_tokens -> inputs_embeds; vision_encoder(pixel_values) -> image_features.
      3. splice image_features into inputs_embeds at image-token positions.
      4. prefill decoder, then greedy-loop with KV cache to max_tokens (early-stop on EOS).
    """
    import numpy as np

    sess = _state["model"]
    tokenizer = _state["tokenizer"]
    image_processor = _state["image_processor"]
    cfg = _state["onnx_cfg"]
    vis, emb, dec = sess["vis"], sess["emb"], sess["dec"]
    feat_np = np.float16 if cfg["feat_dtype"] == "float16" else np.float32
    kv_np = np.float16 if cfg["kv_dtype"] == "float16" else np.float32
    n_layers, n_kv, hd = cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"]
    img_tok = cfg["image_token"]

    # 1) Preprocess the image -> pixel_values, run the vision encoder to learn how
    # many image tokens it produces (FastVLM downsamples by 64 -> ~256 for 1024px).
    pix = image_processor(images=image, return_tensors="np")["pixel_values"]
    pixel_values = np.asarray(pix).astype(feat_np)
    image_features = vis.run(["image_features"], {"pixel_values": pixel_values})[0].astype(feat_np)
    n_img = int(image_features.reshape(-1, image_features.shape[-1]).shape[0])

    # 2) Build the Qwen2 chat prompt with exactly n_img image placeholder tokens
    # spliced in. Qwen2 (llava) format: system + user turn with <image> block.
    user_text = text_prompt or "Describe the image."
    pre = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
           "<|im_start|>user\n")
    post = f"\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    pre_ids = tokenizer(pre, add_special_tokens=False)["input_ids"]
    post_ids = tokenizer(post, add_special_tokens=False)["input_ids"]
    ids = pre_ids + [img_tok] * n_img + post_ids
    input_ids = np.array([ids], dtype=np.int64)

    # 3) embeddings, then splice vision features at the image-token positions
    inputs_embeds = emb.run(["inputs_embeds"], {"input_ids": input_ids})[0].astype(feat_np)
    mask = input_ids[0] == img_tok
    feats = image_features.reshape(-1, image_features.shape[-1])
    inputs_embeds[0, mask, :] = feats[: int(mask.sum())].astype(feat_np)

    seq_len = inputs_embeds.shape[1]
    eos_id = tokenizer.eos_token_id

    # 4) prefill + greedy loop
    past = {f"past_key_values.{i}.{kind}": np.zeros((1, n_kv, 0, hd), dtype=kv_np)
            for i in range(n_layers) for kind in ("key", "value")}
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64)[None, :]
    cur_embeds = inputs_embeds
    generated: list[int] = []

    for step in range(max_tokens):
        feeds = {"inputs_embeds": cur_embeds, "attention_mask": attention_mask,
                 "position_ids": position_ids}
        feeds.update(past)
        out_names = ["logits"] + [f"present.{i}.{k}" for i in range(n_layers) for k in ("key", "value")]
        outputs = dec.run(out_names, feeds)
        logits = outputs[0]
        next_id = int(np.argmax(logits[0, -1, :]))
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        # roll KV cache forward
        for idx, name in enumerate(out_names[1:]):
            layer = name.split(".")[1]
            kind = name.split(".")[2]
            past[f"past_key_values.{layer}.{kind}"] = outputs[idx + 1]
        # next step: single token
        nid = np.array([[next_id]], dtype=np.int64)
        cur_embeds = emb.run(["inputs_embeds"], {"input_ids": nid})[0].astype(feat_np)
        total = attention_mask.shape[1] + 1
        attention_mask = np.ones((1, total), dtype=np.int64)
        position_ids = np.array([[total - 1]], dtype=np.int64)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


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


def _infer_llama(text_prompt: str, image, max_tokens: int) -> str:
    """Run Llama-3.2-11B-Vision-Instruct (4-bit) via its chat template.

    The image is passed as an {"type":"image"} content part; the processor's
    chat template expands it to the <|image|> tile token. We extract {name,symbol}
    of a meme token, matching the contract of the other vision backends — the
    server's _coerce_json then cleans the output into strict JSON.
    """
    import torch

    model = _state["model"]
    processor = _state["image_processor"]

    prompt = text_prompt or (
        "Look at the image. Respond with ONLY a JSON object "
        '{"name": <main subject or prominent text>, '
        '"symbol": <uppercase 3-10 char ticker derived from name>}. '
        "No prose, no markdown."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            temperature=None, top_p=None,
        )
    gen_only = out_ids[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(gen_only, skip_special_tokens=True)
    return text.strip()


# ── PaddleOCR pipeline ─────────────────────────────────────────────────
_OCR_WATERMARKS = (
    "pump.fun", "pumpfun", "dexscreener", "dex screener", ".fun",
    "twitter", "x.com", "t.me", "telegram", "@", "http", "www.",
    "raydium", "jupiter", "solscan", "birdeye", "photon", "bullx",
)


def _run_ocr(image) -> list[tuple[str, float, float]]:
    """Run PaddleOCR. Returns [(text, bbox_height, confidence), ...].

    bbox_height (in px) is used downstream to pick the most prominent line as
    the token name. Handles both the legacy .ocr(np_img) list-of-lists return
    and the 3.x predict() dict return.
    """
    import numpy as np

    ocr = _state["model"]
    arr = np.asarray(image.convert("RGB"))  # PaddleOCR wants BGR/np or path
    arr = arr[:, :, ::-1]  # RGB -> BGR

    results: list[tuple[str, float, float]] = []
    raw = None
    # Legacy 2.x API: ocr.ocr(img, cls=True) -> [[ [box, (text, conf)], ... ]]
    try:
        raw = ocr.ocr(arr, cls=True)
    except TypeError:
        raw = ocr.ocr(arr)

    if not raw:
        return results
    page = raw[0] if (isinstance(raw, list) and raw and isinstance(raw[0], list)) else raw
    if not page:
        return results
    for line in page:
        try:
            box, (text, conf) = line[0], line[1]
            ys = [pt[1] for pt in box]
            height = float(max(ys) - min(ys))
            results.append((str(text).strip(), height, float(conf)))
        except Exception:
            continue
    return results


def ocr_to_token(results: list[tuple[str, float, float]]) -> dict | None:
    """Parse OCR lines into {name, symbol}.

    - symbol: first $TICKER found via regex (uppercase letters, 3-10).
    - name: the largest non-watermark text line (by bbox height).
    - if no explicit ticker, derive symbol from name.
    Returns None if no usable text.
    """
    if not results:
        return None

    def _is_watermark(t: str) -> bool:
        low = t.lower()
        return any(w in low for w in _OCR_WATERMARKS)

    # 1) symbol from $TICKER anywhere in the text
    symbol = ""
    for text, _h, _c in results:
        m = re.search(r"\$([A-Za-z][A-Za-z0-9]{1,9})", text)
        if m:
            symbol = m.group(1).upper()
            break

    # 2) name = largest non-watermark line; strip a leading $TICKER token
    candidates = [
        (text, h) for text, h, _c in results
        if text and not _is_watermark(text)
    ]
    name = ""
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        name = candidates[0][0]
        # drop a standalone $TICKER line being picked as the name
        if re.fullmatch(r"\$[A-Za-z0-9]{2,10}", name.strip()) and len(candidates) > 1:
            name = candidates[1][0]
    if not name:
        # fall back to any text at all
        name = next((t for t, _h, _c in results if t), "")

    name = name.strip()
    if not name and not symbol:
        return None
    if not symbol:
        symbol = _derive_symbol(name)
    return {"name": name, "symbol": symbol}


def _infer_ocr(text_prompt: str, image, max_tokens: int) -> str:
    """OCR backend: read text deterministically, parse to {name,symbol} JSON."""
    results = _run_ocr(image)
    token = ocr_to_token(results)
    if token is None:
        return json.dumps({"name": "", "symbol": ""})
    return json.dumps(token)


def _run_inference(text_prompt: str, image, max_tokens: int) -> tuple[str, float]:
    """Dispatch to the active backend. Returns (raw_text, inference_ms)."""
    t0 = time.perf_counter()
    if _state["backend"] == "ocr":
        raw = _infer_ocr(text_prompt, image, max_tokens)
    elif _state["backend"] == "onnx":
        raw = _infer_onnx(text_prompt, image, max_tokens)
    elif _state["backend"] == "lfm2":
        raw = _infer_lfm2(text_prompt, image, max_tokens)
    elif _state["backend"] == "llama":
        raw = _infer_llama(text_prompt, image, max_tokens)
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
    parsed = False
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            name = str(obj.get("name", "") or "").strip()
            symbol = str(obj.get("symbol", "") or "").strip()
            parsed = True
        except Exception:
            pass
    # Only fall back to raw text as the name when we couldn't parse JSON at all.
    # (A parsed-but-empty name means the backend legitimately found no text —
    # e.g. OCR on an image with no readable token; don't echo the JSON back.)
    if not name and not parsed:
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
