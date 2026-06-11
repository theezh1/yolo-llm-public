"""
local_vision — client for a local OpenAI-compatible vision server (FastVLM).

Same signature as bot.groq_extract_vision: image_source (URL str or bytes) +
prompt → ({name, symbol} | None, elapsed_ms).

The server is expected to expose POST {LOCAL_VISION_URL}
(default http://localhost:8100/v1/chat/completions) speaking the OpenAI
chat-completions schema, including image_url content parts.
"""
import os
import json
import time
import base64
import logging

import httpx

log = logging.getLogger("yolo-llm-bot.local_vision")

LOCAL_VISION_URL = os.environ.get(
    "LOCAL_VISION_URL", "http://localhost:8100/v1/chat/completions"
)
LOCAL_VISION_MODEL = os.environ.get("LOCAL_VISION_MODEL", "apple/FastVLM-0.5B")
LOCAL_VISION_TIMEOUT = float(os.environ.get("LOCAL_VISION_TIMEOUT", "60.0"))


async def local_extract_vision(
    image_source: "bytes | str", prompt: str
) -> tuple[dict | None, float]:
    """Send one image + prompt to the local vision server → {name, symbol}.

    `image_source`:
      - str (URL): forwarded as image_url.url; the server downloads it.
      - bytes: base64 data URL.
    """
    if isinstance(image_source, str):
        image_url_payload = {"url": image_source}
    else:
        b64 = base64.b64encode(image_source).decode()
        image_url_payload = {"url": f"data:image/jpeg;base64,{b64}"}

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=LOCAL_VISION_TIMEOUT) as client:
            r = await client.post(
                LOCAL_VISION_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "model": LOCAL_VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": image_url_payload},
                            ],
                        },
                    ],
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "response_format": {"type": "json_object"},
                },
            )
    except Exception as e:
        log.warning(f"local-vision request err: {e!r}")
        return None, (time.perf_counter() - t0) * 1000

    ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        log.warning(f"local-vision HTTP {r.status_code}: {r.text[:200]}")
        return None, ms
    try:
        raw = r.json()["choices"][0]["message"]["content"]
        return _parse_json_loose(raw), ms
    except Exception as e:
        log.warning(f"local-vision parse err: {e}")
        return None, ms


def _parse_json_loose(raw: str) -> dict | None:
    """Parse model output to {name, symbol}. Tolerates code fences / surrounding text."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        # strip ```json ... ``` fences
        raw = raw.split("```", 2)
        raw = raw[1] if len(raw) > 1 else ""
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # last resort: grab first {...} block
    import re
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
