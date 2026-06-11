"""
yolo-llm-public — Vision-LLM + YOLO crop Telegram bot for meme tokens.

Flow:
  User sends an image (photo, or pbs.twimg.com / *.jpg URL) →
    1. YOLO11n detects objects, picks the "main" one (largest+center heuristic)
    2. Square-crops the main object with padding
    3. A vision LLM extracts {name, symbol} of a meme token from the image
       (or text-extract from a /ai caption)
    4. DexScreener lookup by extracted name
    5. Replies with crop + name/symbol + DexScreener matches + timings

VISION_BACKEND env switch:
  - groq  (default) — Groq vision API (Llama 4 Scout)
  - local           — local OpenAI-compatible vision server (FastVLM), see vision_server/

Run:
  pip install -r requirements.txt
  export BOT_TOKEN=...            # required, no default
  export GROQ_API_KEY=...         # required when VISION_BACKEND=groq
  python bot.py
"""
import io
import os
import re
import sys
import json
import time
import logging
import asyncio

from dotenv import load_dotenv
load_dotenv()

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from ultralytics import YOLO

# ── config ─────────────────────────────────────────────────────────────
# BOT_TOKEN is required — no hardcoded default. Fail fast if missing.
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# Chat whitelist. If ALLOWED_CHAT_ID is 0 (default) the bot serves ALL chats
# (handy for first-run testing). Set it to your group id to lock the bot down.
ALLOWED_CHAT_ID = int(os.environ.get("ALLOWED_CHAT_ID", "0"))

YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolo11n.pt")  # 6 MB nano
CONF_THRESHOLD = float(os.environ.get("YOLO_CONF", "0.25"))
PADDING_PCT = float(os.environ.get("CROP_PADDING_PCT", "0.10"))  # 10% padding around box
PREFER_CLASS = os.environ.get("YOLO_PREFER_CLASS", "person")  # prefer person if present
CROP_OUT_SIZE = int(os.environ.get("CROP_OUT_SIZE", "512"))  # square crops resized to NxN

# Vision backend switch: "groq" (API) or "local" (FastVLM server).
VISION_BACKEND = os.environ.get("VISION_BACKEND", "groq").strip().lower()
# Local model id — used only for the timing-line label (actual call is in local_vision.py).
LOCAL_VISION_MODEL = os.environ.get("LOCAL_VISION_MODEL", "local-vision")

# Groq API for name/symbol extraction (text + vision).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

VISION_PROMPT = (
    'Return ONLY JSON {"name":"...","symbol":"..."} for a pump.fun meme token '
    "from this image. name=Title Case 1-3 words; symbol=UPPERCASE 3-10 chars."
)
GROQ_PROMPT = """You extract pump.fun meme-token name and symbol VERBATIM from a tweet.
Output ONLY JSON: {"name":"...","symbol":"..."}

CRITICAL RULES — preserve, do NOT rewrite:
1. If tweet has $TICKER → that IS the symbol. Use as-is.
2. If tweet has a clear meme phrase (e.g. "fat alex", "shit rocket", "HE KNOWS"),
   that phrase IS the name. Capitalize it. Do NOT add words like "Saga", "Insider", "Launch".
3. symbol: derive from the meme name — uppercase, concatenated, letters only,
   3-10 chars (e.g. "fat alex" → FATALEX, "HE KNOWS" → HEKNOWS, "shit rocket" → SHITROCKET)
4. If tweet is about a general topic with no obvious meme phrase
   (e.g. "FIFA World Cup 2026 starts soon"), pick the most concrete noun phrase
   (e.g. "World Cup 2026" / WORLDCUP).
5. NEVER add creative suffixes. NEVER paraphrase. NEVER add nouns not in tweet.
6. name: Title Case, max 30 chars
7. symbol: ALL CAPS, no special chars, 3-10 chars

Output ONLY the JSON, nothing else."""

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("yolo-llm-bot")

# ── env validation ─────────────────────────────────────────────────────
if not BOT_TOKEN:
    log.error("BOT_TOKEN env var is required and not set. Refusing to start.")
    sys.exit(1)

if ALLOWED_CHAT_ID == 0:
    log.warning(
        "[init] ALLOWED_CHAT_ID=0 — bot will respond in ALL chats. "
        "Set ALLOWED_CHAT_ID to your group id to lock it down."
    )
else:
    log.info(f"[init] ALLOWED_CHAT_ID={ALLOWED_CHAT_ID}")

log.info(f"[init] VISION_BACKEND={VISION_BACKEND}")
if VISION_BACKEND == "groq" and not GROQ_API_KEY:
    log.warning("[init] VISION_BACKEND=groq but GROQ_API_KEY is empty — vision extraction will fail.")

# Local vision client (imported lazily-safe at module load).
from local_vision import local_extract_vision  # noqa: E402

# ── model ──────────────────────────────────────────────────────────────
log.info(f"Loading YOLO model: {YOLO_MODEL}")
model = YOLO(YOLO_MODEL)
log.info("YOLO model loaded")


# ── helpers ────────────────────────────────────────────────────────────
def pick_best_box(boxes: list[dict], img_w: int, img_h: int) -> dict | None:
    """Pick the main object. Heuristic:
      1. If PREFER_CLASS (person) with conf > 0.5 exists — largest of those
      2. Otherwise — largest of all
      3. Tie on area (±5%) — closer to center
    """
    if not boxes:
        return None
    img_cx, img_cy = img_w / 2, img_h / 2

    def area(b):
        return (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

    def center_dist(b):
        cx = (b["x1"] + b["x2"]) / 2
        cy = (b["y1"] + b["y2"]) / 2
        return ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5

    preferred = [b for b in boxes if b["cls_name"] == PREFER_CLASS and b["conf"] >= 0.5]
    candidates = preferred if preferred else boxes
    candidates_sorted = sorted(candidates, key=lambda b: (-area(b), center_dist(b)))
    return candidates_sorted[0]


def square_crop(img: Image.Image, box: dict, padding_pct: float = 0.10,
                out_size: int = 512) -> Image.Image:
    """Square crop around box with padding. Clamps to image bounds."""
    W, H = img.size
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh) * (1.0 + padding_pct * 2)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = side / 2
    sx1, sy1 = cx - half, cy - half
    sx2, sy2 = cx + half, cy + half
    sx1 = max(0, int(sx1)); sy1 = max(0, int(sy1))
    sx2 = min(W, int(sx2)); sy2 = min(H, int(sy2))
    return img.crop((sx1, sy1, sx2, sy2))


def annotate(img: Image.Image, boxes: list[dict], best_idx: int | None) -> Image.Image:
    """Draw all boxes; main one — bright green, others — yellow."""
    out = img.convert("RGB").copy()
    drw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, b in enumerate(boxes):
        is_best = (i == best_idx)
        color = (0, 255, 0) if is_best else (255, 220, 0)
        width = 4 if is_best else 2
        drw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=width)
        label = f"{b['cls_name']} {b['conf']:.2f}"
        if is_best:
            label = "* " + label
        tw = drw.textlength(label, font=font)
        th = 18
        ty = max(0, b["y1"] - th)
        drw.rectangle([b["x1"], ty, b["x1"] + tw + 8, ty + th], fill=(0, 0, 0))
        drw.text((b["x1"] + 4, ty + 1), label, fill=color, font=font)
    return out


def _query_keywords(query: str) -> list[str]:
    """Split query into keywords (lowercase, > 2 chars)."""
    return [w for w in re.split(r"[\s_\-]+", query.lower()) if len(w) > 2]


def _pair_matches_query(p: dict, keywords: list[str]) -> bool:
    """All keywords must appear as WHOLE words in name+symbol (anti substring noise)."""
    bt = p.get("baseToken") or {}
    haystack = (str(bt.get("name", "")) + " " + str(bt.get("symbol", ""))).lower()
    return all(re.search(rf"\b{re.escape(kw)}\b", haystack) for kw in keywords)


async def dexscreener_search(query: str, top_k: int = 3) -> tuple[list[dict], float]:
    """Search DexScreener API, filter by keyword match, dedupe by address, top-K by mcap."""
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.dexscreener.com/latest/dex/search",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
            )
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return [], ms
        pairs = r.json().get("pairs") or []
    except Exception as e:
        log.warning(f"dexscreener err: {e}")
        return [], (time.perf_counter() - t0) * 1000

    def mcap(p: dict) -> float:
        return float(p.get("marketCap") or p.get("fdv") or 0)

    keywords = _query_keywords(query)
    relevant = [p for p in pairs if _pair_matches_query(p, keywords)] if keywords else pairs

    by_token: dict[str, dict] = {}
    for p in relevant:
        bt = p.get("baseToken") or {}
        addr = bt.get("address") or ""
        if not addr:
            continue
        prev = by_token.get(addr)
        if prev is None or mcap(p) > mcap(prev):
            by_token[addr] = p

    return sorted(by_token.values(), key=mcap, reverse=True)[:top_k], ms


async def groq_extract(tweet_text: str) -> tuple[dict | None, float]:
    """Groq Llama 3.1 8B (text-only) → {name, symbol}."""
    if not GROQ_API_KEY:
        return None, 0.0
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": GROQ_PROMPT},
                    {"role": "user", "content": tweet_text},
                ],
                "temperature": 0.3,
                "max_tokens": 60,
                "response_format": {"type": "json_object"},
            },
        )
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        log.warning(f"groq HTTP {r.status_code}: {r.text[:200]}")
        return None, ms
    try:
        raw = r.json()["choices"][0]["message"]["content"]
        return json.loads(raw), ms
    except Exception as e:
        log.warning(f"groq parse err: {e}")
        return None, ms


async def groq_extract_vision(image_source: "bytes | str") -> tuple[dict | None, float]:
    """Groq Vision (Llama 4 Scout) on one image → {name, symbol}.

    `image_source`:
      - bytes: encode → data URL.
      - str (URL): pass URL straight to Groq (it downloads it itself).
    """
    if not GROQ_API_KEY:
        return None, 0.0
    if isinstance(image_source, str):
        image_url_payload = {"url": image_source}
    else:
        import base64
        b64 = base64.b64encode(image_source).decode()
        image_url_payload = {"url": f"data:image/jpeg;base64,{b64}"}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": image_url_payload},
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 30,
                "response_format": {"type": "json_object"},
            },
        )
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        log.warning(f"groq-vision HTTP {r.status_code}: {r.text[:200]}")
        return None, ms
    try:
        raw = r.json()["choices"][0]["message"]["content"]
        return json.loads(raw), ms
    except Exception as e:
        log.warning(f"groq-vision parse err: {e}")
        return None, ms


async def extract_vision(image_source: "bytes | str") -> tuple[dict | None, float]:
    """Dispatch vision extraction to the configured backend."""
    if VISION_BACKEND == "local":
        return await local_extract_vision(image_source, prompt=VISION_PROMPT)
    return await groq_extract_vision(image_source)


def run_yolo(img: Image.Image) -> list[dict]:
    """Run YOLO inference → list of {x1,y1,x2,y2,conf,cls_id,cls_name}."""
    arr = np.array(img.convert("RGB"))
    result = model.predict(arr, conf=CONF_THRESHOLD, verbose=False)[0]
    out: list[dict] = []
    names = result.names
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0].cpu().item())
        conf = float(box.conf[0].cpu().item())
        out.append({
            "x1": float(xyxy[0]),
            "y1": float(xyxy[1]),
            "x2": float(xyxy[2]),
            "y2": float(xyxy[3]),
            "cls_id": cls_id,
            "cls_name": str(names.get(cls_id, str(cls_id))),
            "conf": conf,
        })
    return out


# ── tg handlers ────────────────────────────────────────────────────────
def _chat_allowed(update: Update) -> bool:
    """Guard: True if ALLOWED_CHAT_ID==0 (all chats) or update is from that chat."""
    chat = update.effective_chat
    user = update.effective_user
    chat_id = chat.id if chat else None
    user_id = user.id if user else None
    if ALLOWED_CHAT_ID == 0 or chat_id == ALLOWED_CHAT_ID:
        return True
    log.info(f"[skip] non-allowed chat={chat_id} user={user_id}")
    return False


def _extract_image_url(text: str | None) -> str | None:
    """Extract an image URL from text. Supports direct extensions and pbs.twimg.com."""
    if not text:
        return None
    patterns = [
        r"https?://pbs\.twimg\.com/media/\S+",
        r"https?://\S+\.(?:jpg|jpeg|png|webp|gif)(?:\?\S*)?",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _chat_allowed(update):
        return
    await update.message.reply_text(
        "YOLO + Vision-LLM crop bot\n\n"
        "Commands:\n"
        "  photo (no command) -> /crop = annotation + square crop\n"
        "  /ai <text> + photo -> crop + LLM extracts name/symbol from text\n"
        "  image URL in a message -> vision LLM extracts name/symbol\n"
        "\n"
        f"Config: yolo={YOLO_MODEL} conf>={CONF_THRESHOLD} padding={int(PADDING_PCT*100)}%\n"
        f"        vision_backend={VISION_BACKEND} text_model={GROQ_MODEL}"
    )


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _chat_allowed(update):
        return
    msg = update.message
    if not msg:
        return

    cap = (msg.caption or "").strip().lower()
    if cap.startswith("/ai"):
        return await handle_ai(update, ctx)

    photo_msg = msg
    if not photo_msg.photo and msg.reply_to_message and msg.reply_to_message.photo:
        photo_msg = msg.reply_to_message
    if not photo_msg.photo:
        await msg.reply_text("Send an image (or reply /crop to a message with a photo).")
        return

    photo = photo_msg.photo[-1]
    file = await ctx.bot.get_file(photo.file_id)
    img_bytes = await file.download_as_bytearray()
    try:
        img = Image.open(io.BytesIO(bytes(img_bytes))).convert("RGB")
    except Exception as e:
        await msg.reply_text(f"Could not open image: {e}")
        return

    W, H = img.size
    log.info(f"Photo received: {W}x{H} from chat={msg.chat_id}")

    t0 = time.perf_counter()
    boxes = run_yolo(img)
    yolo_ms = (time.perf_counter() - t0) * 1000
    log.info(f"YOLO: {len(boxes)} objects in {yolo_ms:.1f}ms")

    if not boxes:
        await msg.reply_text(
            f"YOLO found nothing (conf>={CONF_THRESHOLD}).\n"
            f"inference: {yolo_ms:.0f}ms · img: {W}x{H}",
        )
        return

    best = pick_best_box(boxes, W, H)
    best_idx = boxes.index(best) if best else None

    annotated = annotate(img, boxes, best_idx)
    buf_a = io.BytesIO(); annotated.save(buf_a, format="JPEG", quality=88); buf_a.seek(0)

    crop = square_crop(img, best, padding_pct=PADDING_PCT)
    buf_c = io.BytesIO(); crop.save(buf_c, format="JPEG", quality=92); buf_c.seek(0)

    classes_summary = ", ".join(f"{b['cls_name']}({b['conf']:.2f})" for b in boxes[:10])
    if len(boxes) > 10:
        classes_summary += f" ... +{len(boxes)-10}"

    caption_a = (
        f"<b>{len(boxes)} objects</b> · <i>{yolo_ms:.0f}ms</i>\n"
        f"all: {classes_summary}\n"
        f"best: <b>{best['cls_name']}</b> conf={best['conf']:.2f}"
    )
    caption_c = (
        f"<b>square crop</b>\n"
        f"class: <b>{best['cls_name']}</b> · conf {best['conf']:.2f}\n"
        f"size: {crop.size[0]}x{crop.size[1]} · padding {int(PADDING_PCT*100)}%"
    )

    await msg.reply_photo(buf_a, caption=caption_a, parse_mode="HTML")
    await msg.reply_photo(buf_c, caption=caption_c, parse_mode="HTML")


async def _process_image_source(
    image_source: "bytes | str",
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    msg,
    user_text: str = "",
):
    """Common pipeline: image -> vision (or text) extract -> dex -> reply."""
    via = "url" if isinstance(image_source, str) else "bytes"
    use_vision = not user_text

    t_start = time.perf_counter()

    if use_vision:
        extract_task = asyncio.create_task(extract_vision(image_source))
    else:
        extract_task = asyncio.create_task(groq_extract(user_text))

    yolo_task = None
    if via == "bytes":
        def yolo_pipeline_bytes() -> tuple[Image.Image, list[dict], float]:
            t0 = time.perf_counter()
            im = Image.open(io.BytesIO(bytes(image_source))).convert("RGB")
            b = run_yolo(im)
            return im, b, (time.perf_counter() - t0) * 1000

        yolo_task = asyncio.create_task(asyncio.to_thread(yolo_pipeline_bytes))
    else:  # url
        async def download_and_yolo() -> "tuple[Image.Image | None, list[dict], float]":
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                    r = await client.get(image_source)
                    if r.status_code != 200:
                        log.warning(f"[url-yolo] download HTTP {r.status_code}")
                        return None, [], (time.perf_counter() - t0) * 1000
                    img_bytes = r.content
            except Exception as e:
                log.warning(f"[url-yolo] download err: {e!r}")
                return None, [], (time.perf_counter() - t0) * 1000

            def _yolo() -> "tuple[Image.Image, list[dict]]":
                im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                b = run_yolo(im)
                return im, b

            try:
                im, b = await asyncio.to_thread(_yolo)
            except Exception as e:
                log.warning(f"[url-yolo] yolo err: {e!r}")
                return None, [], (time.perf_counter() - t0) * 1000
            return im, b, (time.perf_counter() - t0) * 1000

        yolo_task = asyncio.create_task(download_and_yolo())

    extract, extract_ms = await extract_task

    if via == "url" and not extract:
        await msg.reply_text(
            f"Image failed to load / vision failed: {image_source}",
            disable_web_page_preview=True,
        )
        log.info(f"[url-handler] vision failed url={image_source} extract_ms={extract_ms:.0f}")
        return

    dex_query = (extract or {}).get("name") or user_text or ""
    dex_task = asyncio.create_task(dexscreener_search(dex_query, top_k=3)) if dex_query else None

    tasks = []
    if yolo_task is not None:
        tasks.append(("yolo", yolo_task))
    if dex_task is not None:
        tasks.append(("dex", dex_task))

    img, boxes, yolo_ms = None, [], 0.0
    dex_pairs, dex_ms = [], 0.0
    if tasks:
        results = await asyncio.gather(*(t for _, t in tasks))
        for (kind, _), res in zip(tasks, results):
            if kind == "yolo":
                img, boxes, yolo_ms = res
            elif kind == "dex":
                dex_pairs, dex_ms = res

    crop = None
    best = None
    crop_label = ""
    crop_ms = 0.0
    if img is not None:
        W, H = img.size
        t_crop0 = time.perf_counter()
        if boxes:
            best = pick_best_box(boxes, W, H)
            crop = square_crop(img, best, padding_pct=PADDING_PCT)
            crop_label = f"{best['cls_name']} conf={best['conf']:.2f}"
        else:
            crop = img
            crop_label = "no box (full image)"
        crop_ms = (time.perf_counter() - t_crop0) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000

    name = (extract or {}).get("name", "?")
    symbol = (extract or {}).get("symbol", "?")
    mode = "vision" if use_vision else "text"

    # Short label of the model that actually ran the extract, for the timing line.
    if use_vision:
        _m = GROQ_VISION_MODEL if VISION_BACKEND == "groq" else LOCAL_VISION_MODEL
    else:
        _m = GROQ_MODEL
    model_label = _m.split("/")[-1]  # strip org prefix (LiquidAI/…, meta-llama/…)

    def fmt_mcap(v: float) -> str:
        if v >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"${v/1_000:.1f}k"
        return f"${v:.0f}"

    lines = []
    if extract:
        lines.append(f"<b>{name}</b> · <code>{symbol}</code>  <i>({VISION_BACKEND} {mode})</i>")
    else:
        lines.append(f"{VISION_BACKEND} {mode} failed")

    lines.append("")
    if dex_pairs:
        lines.append(f"DexScreener top-{len(dex_pairs)} for <code>{dex_query}</code>:")
        for i, p in enumerate(dex_pairs, 1):
            bt = p.get("baseToken") or {}
            mc = float(p.get("marketCap") or p.get("fdv") or 0)
            sym = bt.get("symbol", "?")
            nm = (bt.get("name", "") or "")[:30]
            chain = p.get("chainId", "")
            addr = bt.get("address", "")[:8]
            lines.append(f"  {i}. <b>{sym}</b> ({nm}) — mcap {fmt_mcap(mc)} · {chain}/{addr}…")
    elif dex_query:
        lines.append(f"DexScreener: 0 relevant for <code>{dex_query}</code>")

    lines.append("")
    lines.append(f"<i>processing: {total_ms:.0f}ms</i> (yolo ∥ (extract→dex)) · <i>via: {via}</i>")
    lines.append(
        f"  yolo: {yolo_ms:.0f}ms · {model_label}: {extract_ms:.0f}ms · dex: {dex_ms:.0f}ms · crop: {crop_ms:.0f}ms"
    )
    if crop_label:
        lines.append(f"  crop: {crop_label}")

    caption = "\n".join(lines)

    if crop is not None:
        buf_c = io.BytesIO()
        crop.save(buf_c, format="JPEG", quality=92)
        buf_c.seek(0)
        await msg.reply_photo(buf_c, caption=caption, parse_mode="HTML")
    else:
        await msg.reply_text(caption, parse_mode="HTML", disable_web_page_preview=True)


async def handle_ai(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """`/ai <text>` with an attached image or as a reply to an image."""
    if not _chat_allowed(update):
        return
    msg = update.message
    if not msg:
        return

    text_arg = ""
    if msg.caption:
        parts = msg.caption.split(maxsplit=1)
        if len(parts) > 1:
            text_arg = parts[1]
    if not text_arg and ctx.args:
        text_arg = " ".join(ctx.args)
    if not text_arg and msg.reply_to_message:
        text_arg = msg.reply_to_message.caption or msg.reply_to_message.text or ""

    photo_msg = msg
    if not photo_msg.photo and msg.reply_to_message and msg.reply_to_message.photo:
        photo_msg = msg.reply_to_message
    if not photo_msg.photo:
        url = _extract_image_url(text_arg)
        if url:
            log.info(f"[ai-handler] using URL from args: {url}")
            await _process_image_source(url, update, ctx, msg, user_text="")
            return
        await msg.reply_text("Need an image. Attach a photo, paste an image URL, or /ai in reply to a photo.")
        return

    photo = photo_msg.photo[-1]
    file = await ctx.bot.get_file(photo.file_id)
    img_bytes = await file.download_as_bytearray()

    await _process_image_source(bytes(img_bytes), update, ctx, msg, user_text=text_arg)


async def handle_url_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Text message containing an image URL → vision extract (no upload)."""
    if not _chat_allowed(update):
        return
    msg = update.message
    if not msg:
        return
    text = (msg.text or "").strip()
    if not text:
        return
    url = _extract_image_url(text)
    if not url:
        log.debug(f"[url-handler] no image URL in text: {text[:60]!r}")
        return
    log.info(f"[url-handler] detected image URL: {url}")

    try:
        await _process_image_source(url, update, ctx, msg, user_text="")
    except Exception as e:
        log.warning(f"[url-handler] pipeline error url={url}: {e!r}")
        try:
            await msg.reply_text(
                f"Error processing URL: {e!s}",
                disable_web_page_preview=True,
            )
        except Exception:
            pass


# ── main ───────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler(["crop", "yolo"], handle_photo))
    app.add_handler(CommandHandler("ai", handle_ai))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url_message))
    log.info("Bot starting (long-polling)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
