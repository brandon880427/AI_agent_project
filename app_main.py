# app_main.py
# -*- coding: utf-8 -*-
import os, sys, time, json, asyncio, base64, io
import binascii
from typing import Any, Dict, Optional, Tuple, List, Set, Deque
from collections import deque
"""Main web app.

This project is intentionally kept minimal:
- ESP32 camera stream in via /ws/camera
- Browser viewers via /ws/viewer
- Poker cards detection (YOLO) via /api/cards/start
- Send current frame to LLM via /api/describe
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
import bridge_io
import threading
import poker_ev
try:
    import yolomedia  # 确保和 app_main.py 同目录，文件名就是 yolomedia.py
except Exception:
    yolomedia = None  # optional; requires mediapipe/ultralytics
# ---- Windows 事件循环策略 ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# LLM (Ollama) settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:latest")  # e.g. llava:latest / llama3.2-vision

# HTTP client for Ollama (async)
import httpx

# ---- Camera streaming knobs (env) ----
VIEWER_SEND_RAW_JPEG = os.getenv("VIEWER_SEND_RAW_JPEG", "1") != "0"  # if possible, forward ESP32 JPEG directly
VIEWER_MAX_FPS = float(os.getenv("VIEWER_MAX_FPS", "12"))  # global cap; 0 disables
VIEWER_SEND_TIMEOUT_SEC = float(os.getenv("VIEWER_SEND_TIMEOUT_SEC", "0.25"))
VIEWER_MAX_CLIENTS = int(os.getenv("VIEWER_MAX_CLIENTS", "8"))  # 0 disables cap

# If ESP32 camera socket stays connected but stops sending frames, force reconnect.
# Default is intentionally relaxed to avoid flapping on slow startups.
ESP32_CAMERA_RX_TIMEOUT_SEC = float(os.getenv("ESP32_CAMERA_RX_TIMEOUT_SEC", "30"))

# If we keep receiving WS messages (e.g. keepalive text) but no actual JPEG frames,
# the UI will appear frozen. Force a reconnect when frames stall.
ESP32_CAMERA_FRAME_TIMEOUT_SEC = float(os.getenv("ESP32_CAMERA_FRAME_TIMEOUT_SEC", "15"))

app = FastAPI()

# ====== 状态与容器 ======
app.mount("/static", StaticFiles(directory="static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)

camera_viewers: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None

# --- Camera RX debug counters (helps diagnose black screen) ---
camera_rx_connected: bool = False
camera_rx_connected_at: float = 0.0
camera_rx_msg_count: int = 0
camera_rx_text_count: int = 0
camera_rx_bytes_count: int = 0
camera_rx_last_msg_t: float = 0.0
camera_rx_last_frame_t: float = 0.0
camera_rx_last_frame_size: int = 0
camera_rx_invalid_jpeg_count: int = 0
camera_rx_last_invalid_jpeg_reason: Optional[str] = None
camera_rx_last_text: Optional[str] = None
camera_rx_last_text_t: float = 0.0

# ===== ESP32 camera tuning (server -> ESP32 over /ws/camera) =====
# Firmware supports: SET:FRAMESIZE=QVGA/VGA/SVGA/XGA, SET:QUALITY=5..40, SET:FPS=0/5..60
ESP32_SEND_CAMERA_TUNING = os.getenv("ESP32_SEND_CAMERA_TUNING", "0") != "0"
ESP32_STREAM_FRAMESIZE = os.getenv("ESP32_STREAM_FRAMESIZE", "VGA").strip().upper()
ESP32_STREAM_QUALITY = int(os.getenv("ESP32_STREAM_QUALITY", "12"))
ESP32_STREAM_FPS = int(os.getenv("ESP32_STREAM_FPS", "12"))

# ===== Vision pre-processing (helps low-light / blurry frames for LLaVA) =====
VISION_PREPROCESS = os.getenv("VISION_PREPROCESS", "1") != "0"
VISION_PREPROCESS_BRIGHTNESS = float(os.getenv("VISION_PREPROCESS_BRIGHTNESS", "1.20"))
VISION_PREPROCESS_CONTRAST = float(os.getenv("VISION_PREPROCESS_CONTRAST", "1.15"))
VISION_PREPROCESS_JPEG_QUALITY = int(os.getenv("VISION_PREPROCESS_JPEG_QUALITY", "80"))
VISION_MAX_SIDE = int(os.getenv("VISION_MAX_SIDE", "512"))
VISION_OLLAMA_READ_TIMEOUT_SEC = float(os.getenv("VISION_OLLAMA_READ_TIMEOUT_SEC", "120"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "120"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# ===== ESP32 HQ snapshot support (SNAP:HQ) =====
VISION_DESCRIBE_USE_HQ_SNAPSHOT = os.getenv("VISION_DESCRIBE_USE_HQ_SNAPSHOT", "0") != "0"
VISION_HQ_SNAPSHOT_TIMEOUT_SEC = float(os.getenv("VISION_HQ_SNAPSHOT_TIMEOUT_SEC", "2.5"))
_esp32_snapshot_active: bool = False
_esp32_last_snapshot_jpeg: Optional[bytes] = None
_esp32_snapshot_evt: Optional[asyncio.Event] = None

# Viewer broadcast: keep only the newest frame (avoid backlog / eventual freeze)
viewer_latest_jpeg: Optional[bytes] = None
viewer_latest_seq: int = 0
viewer_latest_t: float = 0.0
viewer_new_frame_evt: Optional[asyncio.Event] = None
viewer_broadcast_task: Optional[asyncio.Task] = None
_viewer_last_sent_t: float = 0.0

# Vision describe (Ollama) runtime state
_describe_lock = asyncio.Lock()

_ollama_models_cache: Optional[list[str]] = None


async def _ollama_list_models() -> list[str]:
    """Best-effort list of installed Ollama model tags (names)."""
    global _ollama_models_cache
    if _ollama_models_cache is not None:
        return _ollama_models_cache
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)) as client:
            r = await client.get(f"{OLLAMA_URL.rstrip('/')}/api/tags")
            r.raise_for_status()
            j = r.json()
            models: list[str] = []
            for m in (j.get("models") or []):
                if isinstance(m, dict) and m.get("name"):
                    models.append(str(m["name"]))
            _ollama_models_cache = models
            return models
    except Exception:
        _ollama_models_cache = []
        return []


async def _ollama_choose_model(preferred: str, *, prefer_vision: bool = False) -> tuple[Optional[str], Optional[str]]:
    """Choose a usable Ollama model; returns (model, warning)."""
    preferred = (preferred or "").strip()
    warn_prefix = "[AI] (vision)" if prefer_vision else "[AI] (text)"
    models = await _ollama_list_models()
    if not models:
        return None, f"{warn_prefix} Ollama has no models available (or is unreachable)."

    if preferred:
        if preferred in models:
            return preferred, None
        prefix = preferred + ":"
        for m in models:
            if m.startswith(prefix):
                return m, f"{warn_prefix} Requested model '{preferred}' not found; using '{m}'."

    if prefer_vision:
        for m in models:
            lm = m.lower()
            if "llava" in lm or "vision" in lm:
                warn = None
                if preferred:
                    warn = f"{warn_prefix} Requested model '{preferred}' not found; using '{m}'."
                return m, warn

    # Fallback: first available model.
    warn = None
    if preferred:
        warn = f"{warn_prefix} Requested model '{preferred}' not found; using '{models[0]}'."
    return models[0], warn


def _vision_preprocess_jpeg(jpeg_bytes: bytes) -> bytes:
    if not VISION_PREPROCESS:
        return jpeg_bytes
    try:
        from PIL import Image, ImageEnhance, ImageOps

        im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        if VISION_MAX_SIDE and VISION_MAX_SIDE > 0:
            w, h = im.size
            m = max(w, h)
            if m > VISION_MAX_SIDE:
                scale = VISION_MAX_SIDE / float(m)
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                im = im.resize((nw, nh), Image.BILINEAR)
        im = ImageOps.autocontrast(im)
        if abs(VISION_PREPROCESS_BRIGHTNESS - 1.0) > 1e-3:
            im = ImageEnhance.Brightness(im).enhance(VISION_PREPROCESS_BRIGHTNESS)
        if abs(VISION_PREPROCESS_CONTRAST - 1.0) > 1e-3:
            im = ImageEnhance.Contrast(im).enhance(VISION_PREPROCESS_CONTRAST)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=VISION_PREPROCESS_JPEG_QUALITY, optimize=True)
        return out.getvalue()
    except Exception:
        return jpeg_bytes


async def _esp32_send_camera_cmd(cmd: str) -> bool:
    ws = esp32_camera_ws
    if ws is None:
        return False
    try:
        await ws.send_text(cmd)
        return True
    except Exception:
        return False


async def _esp32_request_hq_snapshot() -> Optional[bytes]:
    global _esp32_snapshot_active, _esp32_last_snapshot_jpeg, _esp32_snapshot_evt
    if esp32_camera_ws is None:
        return None
    if _esp32_snapshot_evt is None:
        _esp32_snapshot_evt = asyncio.Event()

    _esp32_last_snapshot_jpeg = None
    _esp32_snapshot_active = True
    _esp32_snapshot_evt.clear()

    ok = await _esp32_send_camera_cmd("SNAP:HQ")
    if not ok:
        _esp32_snapshot_active = False
        return None

    try:
        await asyncio.wait_for(_esp32_snapshot_evt.wait(), timeout=VISION_HQ_SNAPSHOT_TIMEOUT_SEC)
    except Exception:
        pass
    finally:
        _esp32_snapshot_active = False

    return _esp32_last_snapshot_jpeg


def _viewer_set_latest(jpeg_bytes: bytes) -> None:
    global viewer_latest_jpeg, viewer_latest_seq, viewer_latest_t
    if not jpeg_bytes:
        return
    viewer_latest_jpeg = jpeg_bytes
    viewer_latest_seq += 1
    viewer_latest_t = time.monotonic()
    try:
        if viewer_new_frame_evt is not None:
            viewer_new_frame_evt.set()
    except Exception:
        pass


async def _viewer_broadcast_loop():
    global _viewer_last_sent_t
    last_sent_seq = 0
    min_interval = (1.0 / VIEWER_MAX_FPS) if VIEWER_MAX_FPS and VIEWER_MAX_FPS > 0 else 0.0

    async def _safe_send(ws: WebSocket, jpeg: bytes) -> bool:
        try:
            await asyncio.wait_for(ws.send_bytes(jpeg), timeout=VIEWER_SEND_TIMEOUT_SEC)
            return True
        except Exception:
            return False

    while True:
        if viewer_new_frame_evt is None:
            await asyncio.sleep(0.05)
            continue
        await viewer_new_frame_evt.wait()
        viewer_new_frame_evt.clear()

        seq = viewer_latest_seq
        if seq == last_sent_seq:
            continue
        jpeg = viewer_latest_jpeg
        if (not jpeg) or (not camera_viewers):
            last_sent_seq = seq
            continue

        if min_interval > 0.0:
            now = time.monotonic()
            wait = (_viewer_last_sent_t + min_interval) - now
            if wait > 0:
                await asyncio.sleep(wait)
            _viewer_last_sent_t = time.monotonic()

        viewers = list(camera_viewers)
        if viewers:
            results = await asyncio.gather(*(_safe_send(ws, jpeg) for ws in viewers), return_exceptions=True)
            for ws, ok in zip(viewers, results):
                if ok is not True:
                    camera_viewers.discard(ws)

        last_sent_seq = seq



# ============== YOLO媒体线程管理 =================
yolomedia_thread: Optional[threading.Thread] = None
yolomedia_stop_event = threading.Event()
yolomedia_running = False
yolomedia_sending_frames = False  # 新增：标记YOLO是否已经开始发送处理后的帧
_yolomedia_last_frame_t: float = 0.0
YOLO_FRAME_STALE_SEC = float(os.getenv("YOLO_FRAME_STALE_SEC", "2.0"))

async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)

async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[UI FINAL] {text}", flush=True)


class DescribeRequest(BaseModel):
    prompt: Optional[str] = None


class PokerAdviceRequest(BaseModel):
    # If provided, overrides YOLO detections.
    cards: Optional[List[str]] = None  # e.g. ["As", "Kd"]
    num_opponents: int = 1
    iters: int = 2000
    pot: Optional[float] = None
    to_call: Optional[float] = None
    use_llm: bool = True


async def _ollama_describe_image(jpeg_bytes: bytes, prompt: Optional[str] = None) -> str:
    requested = (OLLAMA_VISION_MODEL or OLLAMA_MODEL).strip()
    model, warn = await _ollama_choose_model(requested, prefer_vision=True)
    if warn:
        await ui_broadcast_final(warn)
    if not model:
        raise RuntimeError("Ollama not available or has no models")

    user_prompt = (prompt or "Briefly describe what is in this image (main objects / scene). Use English.").strip()
    jpeg_bytes = _vision_preprocess_jpeg(jpeg_bytes)
    img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

    payload = {
        "model": model,
        "stream": False,
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
        },
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
                "images": [img_b64],
            }
        ],
    }

    timeout = httpx.Timeout(connect=5.0, read=VISION_OLLAMA_READ_TIMEOUT_SEC, write=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OLLAMA_URL.rstrip('/')}/api/chat", json=payload)
        # Some errors return JSON like {"error": "model ... not found"}
        try:
            data = r.json()
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data.get("error")))
        r.raise_for_status()
        if data is None:
            data = r.json()

    # Ollama /api/chat returns {"message": {"content": "..."}, ...}
    content = ""
    try:
        content = (data.get("message") or {}).get("content") or ""
    except Exception:
        content = ""
    content = (content or "").strip()
    return content or "(no description generated)"


async def _ollama_chat_text(prompt: str) -> str:
    """Chat with a text-only model via Ollama."""
    requested = (OLLAMA_MODEL or OLLAMA_VISION_MODEL).strip()
    model, warn = await _ollama_choose_model(requested, prefer_vision=False)
    # Don't broadcast model-fallback warnings to the UI for text chat.
    # (It is confusing during Poker EV; we still log it for debugging.)
    if warn:
        try:
            print(warn, flush=True)
        except Exception:
            pass
    if not model:
        raise RuntimeError("Ollama not available or has no models")

    payload = {
        "model": model,
        "stream": False,
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
        },
        "messages": [
            {
                "role": "user",
                "content": (prompt or "").strip(),
            }
        ],
    }

    timeout = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{OLLAMA_URL.rstrip('/')}/api/chat", json=payload)
        try:
            data = r.json()
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data.get("error")))
        r.raise_for_status()
        if data is None:
            data = r.json()

    content = ""
    try:
        content = (data.get("message") or {}).get("content") or ""
    except Exception:
        content = ""
    return (content or "").strip()


async def _describe_current_frame(prompt: Optional[str] = None) -> None:
    jpeg: Optional[bytes] = None
    if VISION_DESCRIBE_USE_HQ_SNAPSHOT and esp32_camera_ws is not None:
        try:
            await ui_broadcast_final("[AI] (vision) Capturing a high-quality frame...")
            jpeg = await _esp32_request_hq_snapshot()
        except Exception:
            jpeg = None

    # Prefer last_frames because viewer_latest_jpeg is only updated when viewers exist.
    if not jpeg:
        try:
            if last_frames:
                jpeg = last_frames[-1][1]
        except Exception:
            jpeg = None
    if not jpeg:
        jpeg = viewer_latest_jpeg
    if not jpeg:
        await ui_broadcast_final("[AI] (vision) No camera frame received yet; cannot analyze.")
        return

    await ui_broadcast_final("[AI] (vision) Analyzing the current frame...")
    try:
        desc = await _ollama_describe_image(jpeg, prompt=prompt)
        await ui_broadcast_final(f"[AI] (vision) {desc}")
    except httpx.ReadTimeout:
        await ui_broadcast_final(
            "[AI] (vision) Timed out waiting for Ollama. On CPU this can be slow; try smaller frames (VISION_MAX_SIDE=512) or increase VISION_OLLAMA_READ_TIMEOUT_SEC."
        )
    except httpx.HTTPStatusError as e:
        # Often happens when using a non-vision model.
        body = ""
        try:
            body = e.response.text
        except Exception:
            body = ""
        msg = f"Ollama HTTP {e.response.status_code}"
        if body:
            msg += f": {body[:300]}"
        await ui_broadcast_final(f"[AI] (vision) Failed: {msg}")
    except Exception as e:
        err = str(e).strip() or repr(e)
        await ui_broadcast_final(f"[AI] (vision) Failed: {err}")

# ========= 启动/停止 YOLO 媒体处理 =========
def start_cards_yolomedia():
    """Start YOLO poker-cards mode."""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    # 如果已经在运行，先停止
    if yolomedia_running:
        stop_yolomedia()
    
    yolo_class = "cards"
    print(f"[YOLOMEDIA] Starting cards mode: {yolo_class}", flush=True)
    
    yolomedia_stop_event.clear()
    yolomedia_running = True
    yolomedia_sending_frames = False  # 重置发送帧状态
    
    def _run():
        try:
            yolomedia.main(headless=True, prompt_name=yolo_class, stop_event=yolomedia_stop_event)
        except Exception as e:
            print(f"[YOLOMEDIA] worker stopped: {e}", flush=True)
        finally:
            global yolomedia_running, yolomedia_sending_frames
            yolomedia_running = False
            yolomedia_sending_frames = False
    
    yolomedia_thread = threading.Thread(target=_run, daemon=True)
    yolomedia_thread.start()
    print(f"[YOLOMEDIA] background worker started for: {yolo_class}（正在初始化，暂时显示原始画面）", flush=True)

def stop_yolomedia():
    """停止yolomedia线程"""
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    
    if yolomedia_running:
        print("[YOLOMEDIA] Stopping worker...", flush=True)
        yolomedia_stop_event.set()
        
        # 等待线程结束（最多等5秒）
        if yolomedia_thread and yolomedia_thread.is_alive():
            yolomedia_thread.join(timeout=5.0)
        
        yolomedia_running = False
        yolomedia_sending_frames = False

        print("[YOLOMEDIA] Worker stopped.", flush=True)


# ---------- 页面 / 健康 ----------
@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(
            f.read(),
            headers={
                "Cache-Control": "no-store",
            },
        )

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


@app.get("/api/camera/debug")
async def api_camera_debug() -> Dict[str, Any]:
    now = time.monotonic()
    frame_age = (now - camera_rx_last_frame_t) if camera_rx_last_frame_t else None
    msg_age = (now - camera_rx_last_msg_t) if camera_rx_last_msg_t else None
    text_age = (now - camera_rx_last_text_t) if camera_rx_last_text_t else None
    no_frames_yet_age = (now - camera_rx_connected_at) if (camera_rx_connected and (camera_rx_last_frame_t <= 0) and camera_rx_connected_at) else None
    frame_stalled = False
    if ESP32_CAMERA_FRAME_TIMEOUT_SEC and ESP32_CAMERA_FRAME_TIMEOUT_SEC > 0:
        if frame_age is not None and frame_age > ESP32_CAMERA_FRAME_TIMEOUT_SEC:
            frame_stalled = True
        if no_frames_yet_age is not None and no_frames_yet_age > ESP32_CAMERA_FRAME_TIMEOUT_SEC:
            frame_stalled = True
    return {
        "ok": True,
        "connected": camera_rx_connected,
        "connected_for_sec": round(now - camera_rx_connected_at, 3) if camera_rx_connected_at else None,
        "msg_count": camera_rx_msg_count,
        "text_count": camera_rx_text_count,
        "bytes_count": camera_rx_bytes_count,
        "last_msg_age_sec": round(msg_age, 3) if msg_age is not None else None,
        "last_frame_age_sec": round(frame_age, 3) if frame_age is not None else None,
        "last_frame_size": camera_rx_last_frame_size,
        "invalid_jpeg_count": camera_rx_invalid_jpeg_count,
        "last_invalid_jpeg_reason": camera_rx_last_invalid_jpeg_reason,
        "last_text": camera_rx_last_text,
        "last_text_age_sec": round(text_age, 3) if text_age is not None else None,
        "frame_timeout_sec": ESP32_CAMERA_FRAME_TIMEOUT_SEC,
        "frame_stalled": frame_stalled,
        "viewer_clients": len(camera_viewers),
        "viewer_latest_seq": viewer_latest_seq,
        "viewer_latest_age_sec": round(now - viewer_latest_t, 3) if viewer_latest_t else None,
    }


@app.get("/api/viewer/last.jpg")
async def api_viewer_last_jpg() -> Response:
    """Return the latest JPEG that would be sent to /ws/viewer.

    This is useful to verify Cards/YOLO overlays, because /api/camera/last.jpg
    intentionally returns the raw incoming camera frame.
    """
    jpeg = viewer_latest_jpeg
    if not jpeg:
        return Response(status_code=404, content=b"no viewer frame")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/camera/last.jpg")
async def api_camera_last_jpg() -> Response:
    jpeg: Optional[bytes] = None
    if last_frames:
        try:
            _, jpeg = last_frames[-1]
        except Exception:
            jpeg = None
    if jpeg is None:
        jpeg = viewer_latest_jpeg

    if not jpeg:
        return Response(status_code=404, content=b"no frame")

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/cards/start")
async def api_cards_start() -> Dict[str, Any]:
    """Start poker cards recognition (YOLO)."""
    if yolomedia is None:
        return {"ok": False, "error": "yolomedia_unavailable"}

    try:
        start_cards_yolomedia()
        try:
            await ui_broadcast_final("[SYSTEM] Cards recognition started")
        except Exception:
            pass
        return {"ok": True}
    except Exception as e:
        try:
            await ui_broadcast_final(f"[SYSTEM] Cards start failed: {e}")
        except Exception:
            pass
        return {"ok": False, "error": str(e)}


@app.post("/api/cards/stop")
async def api_cards_stop() -> Dict[str, Any]:
    """Stop cards recognition."""
    try:
        if yolomedia_running:
            stop_yolomedia()
        try:
            await ui_broadcast_final("[SYSTEM] Cards recognition stopped")
        except Exception:
            pass
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/cards/dets")
async def api_cards_dets() -> Dict[str, Any]:
    """Return latest cards detections from yolomedia (if available)."""
    if yolomedia is None or not hasattr(yolomedia, "get_latest_cards_dets"):
        return {"ok": False, "error": "yolomedia_unavailable"}
    try:
        st = yolomedia.get_latest_cards_dets()  # type: ignore[attr-defined]
        return {"ok": True, "state": st}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/camera/snap")
async def api_camera_snap() -> Dict[str, Any]:
    jpeg = await _esp32_request_hq_snapshot()
    if jpeg:
        try:
            last_frames.append((time.time(), jpeg))
        except Exception:
            pass
        try:
            _viewer_set_latest(jpeg)
        except Exception:
            pass
        return {"ok": True, "received": True, "size": len(jpeg)}
    return {"ok": True, "received": False, "size": 0}


@app.post("/api/describe")
async def api_describe(req: DescribeRequest):
    # Single-flight: avoid stacking expensive vision requests.
    if _describe_lock.locked():
        return {"ok": False, "status": "busy"}

    async def _run():
        async with _describe_lock:
            await _describe_current_frame(prompt=req.prompt)

    asyncio.create_task(_run())
    return {"ok": True, "status": "accepted"}


@app.post("/api/poker/advice")
async def api_poker_advice(req: PokerAdviceRequest) -> Dict[str, Any]:
    """Texas Hold'em preflop: estimate equity / EV and suggest bet/call/fold.

    Default behavior uses the latest 2 cards from YOLO (cards mode).
    You can override by sending {"cards": ["As","Kd"]}.
    """

    def _pick_two_from_yolo() -> tuple[list[str], dict]:
        if yolomedia is None or not hasattr(yolomedia, "get_latest_cards_dets"):
            return [], {"ok": False, "error": "yolomedia_cards_state_unavailable"}
        st = yolomedia.get_latest_cards_dets()  # type: ignore[attr-defined]

        # Prefer stabilized tracker output when available.
        try:
            hole_cards = list((st or {}).get("hole_cards") or [])
            hole_cards = [str(x).strip() for x in hole_cards if str(x).strip()]
            if len(hole_cards) >= 2:
                chosen = [hole_cards[0], hole_cards[1]]
                return chosen, {"ok": True, "yolo": st, "picked": "hole_cards"}
        except Exception:
            pass

        dets = list((st or {}).get("dets") or [])
        # Keep only parseable card labels.
        cleaned: list[dict] = []
        for d in dets:
            try:
                rank = d.get("rank")
                suit = d.get("suit")
                disp = d.get("display")
                conf = float(d.get("conf") or 0.0)
                if not rank or not suit:
                    continue
                # Convert to compact format like "AS".
                suit_ch = {"spades": "S", "hearts": "H", "diamonds": "D", "clubs": "C"}.get(str(suit), "")
                r = str(rank).upper().replace("10", "T")
                if suit_ch and r:
                    cleaned.append({"card": f"{r}{suit_ch}", "conf": conf, "display": disp, "raw": d})
            except Exception:
                continue
        cleaned.sort(key=lambda x: x.get("conf", 0.0), reverse=True)

        chosen: list[str] = []
        seen: set[str] = set()
        for c in cleaned:
            s = str(c["card"]).upper().strip()
            if s in seen:
                continue
            seen.add(s)
            chosen.append(s)
            if len(chosen) >= 2:
                break
        return chosen, {"ok": True, "yolo": st, "picked": cleaned[:8]}

    # 1) Determine hole cards.
    cards_in: list[str] = []
    if req.cards:
        cards_in = [str(x).strip() for x in req.cards if str(x).strip()]
    else:
        cards_in, yolo_dbg = _pick_two_from_yolo()
        if not cards_in:
            return {
                "ok": False,
                "error": "no_cards_detected",
                "hint": "Start YOLO cards first: POST /api/cards/start, then ensure two cards are visible.",
                "debug": yolo_dbg,
            }

    if len(cards_in) < 2:
        return {"ok": False, "error": "need_two_cards", "cards": cards_in}

    # Normalize formats: allow A♠, As, 10h, etc.
    try:
        hole = [poker_ev.parse_card(cards_in[0]), poker_ev.parse_card(cards_in[1])]
    except Exception as e:
        return {"ok": False, "error": "invalid_cards", "cards": cards_in[:2], "detail": str(e)}

    # 2) Compute equity in a worker thread to avoid blocking the event loop.
    num_opponents = int(req.num_opponents or 1)
    iters = int(req.iters or 2000)
    try:
        equity = await asyncio.to_thread(
            poker_ev.estimate_equity,
            hole,
            num_opponents=num_opponents,
            iters=iters,
        )
    except Exception as e:
        return {"ok": False, "error": "equity_failed", "detail": str(e)}

    advice = poker_ev.advise_action(equity, pot_before_call=req.pot, to_call=req.to_call)

    result: Dict[str, Any] = {
        "ok": True,
        "cards": [poker_ev.format_card(hole[0]), poker_ev.format_card(hole[1])],
        "num_opponents": num_opponents,
        "iters": iters,
        "equity": float(advice.equity),
        "pot_odds": (float(advice.pot_odds) if advice.pot_odds is not None else None),
        "ev_call": (float(advice.ev_call) if advice.ev_call is not None else None),
        "action": advice.action,
        "reason": advice.reason,
    }

    # 3) Ask LLM for a natural-language recommendation (optional).
    if req.use_llm:
        try:
            prompt = (
                "你是德州撲克教練。\n"
                "我手牌是兩張（preflop）: {c1} {c2}\n"
                "注意：請不要把這兩張牌說錯（例如誤說成一對）。\n"
                "對手數量(隨機範圍假設): {n} 人\n"
                "蒙地卡羅估計勝率(=equity): {eq:.3f}\n"
            ).format(c1=result["cards"][0], c2=result["cards"][1], n=num_opponents, eq=result["equity"])
            if req.pot is not None and req.to_call is not None:
                prompt += (
                    f"目前底池(我跟注前): {req.pot}\n"
                    f"我需要跟注金額: {req.to_call}\n"
                    f"估計跟注EV: {result['ev_call']}\n"
                    f"需要勝率(底池賠率門檻): {result['pot_odds']}\n"
                )
            prompt += (
                "\n請用繁體中文回答，給出：\n"
                "1) 這手牌大致強度解讀\n"
                "2) 建議動作：bet / call / raise / fold / check（選一個最適合）\n"
                "3) 用1-3句說明理由（不要太長，不要重複列出數字）\n"
            )

            llm_text = await _ollama_chat_text(prompt)
            result["llm"] = llm_text
        except Exception as e:
            result["llm_error"] = str(e)

    return result

# ---------- WebSocket：WebUI 文本（ASR/AI 状态推送） ----------
@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        try:
            await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        except Exception:
            return
        while True:
            await asyncio.sleep(60)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        ui_clients.pop(id(ws), None)

# ---------- WebSocket：ESP32 相机入口（JPEG 二进制） ----------
@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    global esp32_camera_ws
    global camera_rx_connected, camera_rx_connected_at, camera_rx_msg_count, camera_rx_text_count
    global camera_rx_bytes_count, camera_rx_last_msg_t, camera_rx_last_frame_t, camera_rx_last_frame_size
    global camera_rx_invalid_jpeg_count, camera_rx_last_invalid_jpeg_reason
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CAMERA] ESP32 connected")

    camera_rx_connected = True
    camera_rx_connected_at = time.monotonic()
    camera_rx_msg_count = 0
    camera_rx_text_count = 0
    camera_rx_bytes_count = 0
    camera_rx_last_msg_t = camera_rx_connected_at
    camera_rx_last_frame_t = 0.0
    camera_rx_last_frame_size = 0
    camera_rx_invalid_jpeg_count = 0
    camera_rx_last_invalid_jpeg_reason = None
    global camera_rx_last_text, camera_rx_last_text_t
    camera_rx_last_text = None
    camera_rx_last_text_t = 0.0

    # Ask ESP32 to use a better streaming profile (firmware supports these commands).
    # Some firmwares/boards can stall when applying framesize changes; allow disabling.
    if ESP32_SEND_CAMERA_TUNING:
        try:
            if ESP32_STREAM_FRAMESIZE:
                await _esp32_send_camera_cmd(f"SET:FRAMESIZE={ESP32_STREAM_FRAMESIZE}")
            if ESP32_STREAM_QUALITY:
                await _esp32_send_camera_cmd(f"SET:QUALITY={int(ESP32_STREAM_QUALITY)}")
            # 0 means unlimited (ESP32 side); otherwise it clamps 5..60.
            await _esp32_send_camera_cmd(f"SET:FPS={int(ESP32_STREAM_FPS)}")
        except Exception:
            pass

    frame_counter = 0  # 添加帧计数器
    t0 = time.monotonic()
    last_log_t = t0
    last_msg_t = time.monotonic()
    last_frame_t = 0.0
    sent_stream_kick = False

    def _try_decode_jpeg_text_payload(text_payload: str) -> Optional[bytes]:
        """Best-effort: accept ESP32 frames sent as base64 text (or JSON/data URL).

        Returns decoded JPEG bytes if it looks like a JPEG, else None.
        """
        if not text_payload:
            return None

        s = text_payload.strip()
        if not s:
            return None

        # data URL: data:image/jpeg;base64,...
        if s.startswith("data:image"):
            comma = s.find(",")
            if comma != -1:
                s = s[comma + 1 :].strip()

        # JSON wrapper: {"jpg":"...base64..."} etc.
        if s.startswith("{") or s.startswith("["):
            try:
                j = json.loads(s)
                if isinstance(j, dict):
                    for k in ("jpg", "jpeg", "frame", "data", "image", "b64"):
                        v = j.get(k)
                        if isinstance(v, str) and v.strip():
                            s = v.strip()
                            break
            except Exception:
                return None

        # Heuristic: short strings are likely control messages, not frames.
        if len(s) < 200:
            return None

        b64 = "".join(s.split())
        # Fix missing padding
        pad = (-len(b64)) % 4
        if pad:
            b64 = b64 + ("=" * pad)

        try:
            raw = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError):
            return None

        # JPEG magic bytes
        if raw.startswith(b"\xff\xd8"):
            return raw
        return None
    
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=ESP32_CAMERA_RX_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                # No WS messages arrived within timeout.
                stall_s = time.monotonic() - last_msg_t
                if last_frame_t:
                    frame_stall_s = time.monotonic() - last_frame_t
                    print(
                        f"[CAMERA] stalled: no ws messages for {stall_s:.1f}s (last_frame_age={frame_stall_s:.1f}s); closing for reconnect",
                        flush=True,
                    )
                else:
                    print(
                        f"[CAMERA] stalled: no ws messages for {stall_s:.1f}s (no frames received yet); closing for reconnect",
                        flush=True,
                    )
                break

            # If we keep receiving WS messages but no frames, consider the camera stalled.
            now_mono = time.monotonic()
            if ESP32_CAMERA_FRAME_TIMEOUT_SEC and ESP32_CAMERA_FRAME_TIMEOUT_SEC > 0:
                if last_frame_t <= 0.0:
                    # No frames received since connect
                    if (now_mono - camera_rx_connected_at) > ESP32_CAMERA_FRAME_TIMEOUT_SEC:
                        stall_s = now_mono - camera_rx_connected_at
                        try:
                            await ui_broadcast_final(
                                f"[SYSTEM] Camera stalled: connected {stall_s:.1f}s but no frames. Reconnecting…"
                            )
                        except Exception:
                            pass
                        print(
                            f"[CAMERA] stalled: connected {stall_s:.1f}s but no frames received; closing for reconnect",
                            flush=True,
                        )
                        break
                else:
                    frame_stall_s = now_mono - last_frame_t
                    if frame_stall_s > ESP32_CAMERA_FRAME_TIMEOUT_SEC:
                        try:
                            await ui_broadcast_final(
                                f"[SYSTEM] Camera stalled: no frames for {frame_stall_s:.1f}s. Reconnecting…"
                            )
                        except Exception:
                            pass
                        print(
                            f"[CAMERA] stalled: no frames for {frame_stall_s:.1f}s (keepalive still arriving); closing for reconnect",
                            flush=True,
                        )
                        break

            camera_rx_msg_count += 1
            last_msg_t = time.monotonic()
            camera_rx_last_msg_t = last_msg_t

            if "text" in msg and msg["text"] is not None:
                camera_rx_text_count += 1
                t = (msg["text"] or "").strip()
                camera_rx_last_text = t[:200] if t else ""
                camera_rx_last_text_t = time.monotonic()

                # Log the first few non-frame text messages; they often reveal firmware state.
                if camera_rx_text_count <= 3 and t and t not in ("SNAP:BEGIN", "SNAP:END"):
                    print(f"[CAMERA] text msg: {t[:120]}", flush=True)
                if t == "SNAP:BEGIN":
                    # Next binary frame is the HQ snapshot (then SNAP:END)
                    global _esp32_snapshot_active, _esp32_last_snapshot_jpeg
                    _esp32_snapshot_active = True
                    _esp32_last_snapshot_jpeg = None
                    continue
                if t == "SNAP:END":
                    global _esp32_snapshot_evt
                    _esp32_snapshot_active = False
                    if _esp32_snapshot_evt is not None:
                        try:
                            _esp32_snapshot_evt.set()
                        except Exception:
                            pass
                    continue

                # Some ESP32 firmwares send JPEG frames as base64 text.
                decoded = _try_decode_jpeg_text_payload(t)
                if decoded is not None:
                    msg["bytes"] = decoded
                else:
                    # If we are receiving keepalive/control text but no frames yet, try a one-time "kick"
                    # to start streaming (some firmwares only begin sending frames after a setting command).
                    if (not sent_stream_kick) and (last_frame_t <= 0.0):
                        now_kick = time.monotonic()
                        if (now_kick - camera_rx_connected_at) >= 1.0:
                            sent_stream_kick = True
                            try:
                                if ESP32_STREAM_FRAMESIZE:
                                    await _esp32_send_camera_cmd(f"SET:FRAMESIZE={ESP32_STREAM_FRAMESIZE}")
                                if ESP32_STREAM_QUALITY:
                                    await _esp32_send_camera_cmd(f"SET:QUALITY={int(ESP32_STREAM_QUALITY)}")
                                await _esp32_send_camera_cmd(f"SET:FPS={int(ESP32_STREAM_FPS)}")
                                print("[CAMERA] sent stream kick (SET:*)", flush=True)
                            except Exception:
                                pass
                    # Ignore non-frame text messages.
                    continue

            if "bytes" in msg and msg["bytes"] is not None:
                camera_rx_bytes_count += 1
                data = msg["bytes"]

                # JPEG validity check (cheap marker heuristics):
                # - Must start with SOI (FFD8) and end with EOI (FFD9)
                # - Must contain SOF (FFC0/FFC2) + SOS (FFDA)
                # - Must contain quantization (FFDB) + Huffman tables (FFC4)
                # Missing any of these often triggers "JPEG decode failed" on the browser.
                try:
                    has_soi = data.startswith(b"\xff\xd8")
                    has_eoi = data.endswith(b"\xff\xd9")
                    has_sof = (b"\xff\xc0" in data) or (b"\xff\xc2" in data)
                    has_sos = (b"\xff\xda" in data)
                    has_dqt = (b"\xff\xdb" in data)
                    has_dht = (b"\xff\xc4" in data)
                    if not (has_soi and has_eoi and has_sof and has_sos and has_dqt and has_dht):
                        missing: list[str] = []
                        if not has_soi:
                            missing.append("SOI")
                        if not has_eoi:
                            missing.append("EOI")
                        if not has_sof:
                            missing.append("SOF")
                        if not has_sos:
                            missing.append("SOS")
                        if not has_dqt:
                            missing.append("DQT")
                        if not has_dht:
                            missing.append("DHT")
                        camera_rx_invalid_jpeg_count += 1
                        camera_rx_last_invalid_jpeg_reason = "missing " + "+".join(missing)
                        camera_rx_last_frame_size = int(len(data))
                        continue
                except Exception:
                    # If validation fails unexpectedly, do not block the stream.
                    pass

                last_frame_t = time.monotonic()
                camera_rx_last_frame_t = last_frame_t
                camera_rx_last_frame_size = int(len(data))

                # If we're in a SNAP:HQ window, capture this JPEG as the snapshot.
                if _esp32_snapshot_active:
                    try:
                        _esp32_last_snapshot_jpeg = data
                        # Also treat snapshot as latest frame for viewer/agent.
                        last_frames.append((time.time(), data))
                        if camera_viewers:
                            _viewer_set_latest(data)
                    except Exception:
                        pass

                frame_counter += 1

                if frame_counter == 1:
                    print(f"[CAMERA] first frame received: {len(data)} bytes", flush=True)
                now_t = time.monotonic()
                if (frame_counter % 150) == 0 and (now_t - last_log_t) > 0.1:
                    fps = 150.0 / (now_t - last_log_t)
                    print(f"[CAMERA] recv fps≈{fps:.1f} (frame={frame_counter})", flush=True)
                    last_log_t = now_t
                
                try:
                    last_frames.append((time.time(), data))
                except Exception:
                    pass
                
                # 推送到bridge_io（供yolomedia使用）
                bridge_io.push_raw_jpeg(data)

                # When YOLO (Cards) is actively sending frames, let yolomedia own the viewer.
                # Otherwise forward the raw ESP32 JPEG to the viewer.
                yolo_active = bool(
                    yolomedia_sending_frames
                    and (_yolomedia_last_frame_t > 0.0)
                    and ((time.monotonic() - _yolomedia_last_frame_t) <= YOLO_FRAME_STALE_SEC)
                )
                if (not yolo_active) and camera_viewers:
                    try:
                        if VIEWER_SEND_RAW_JPEG:
                            _viewer_set_latest(data)
                        else:
                            arr = np.frombuffer(data, dtype=np.uint8)
                            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if bgr is not None and bgr.size > 0:
                                ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                if ok:
                                    _viewer_set_latest(enc.tobytes())
                    except Exception as e:
                        if frame_counter % 60 == 0:
                            print(f"[CAMERA] Broadcast error: {e}")

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        camera_rx_connected = False
        print("[CAMERA] ESP32 disconnected")

# ---------- WebSocket：浏览器订阅相机帧 ----------
@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()

    # Optional guardrail: too many viewer sockets can stall broadcast and look like a frozen camera.
    if VIEWER_MAX_CLIENTS and VIEWER_MAX_CLIENTS > 0 and len(camera_viewers) >= VIEWER_MAX_CLIENTS:
        try:
            await ws.close(code=1013)
        except Exception:
            pass
        return

    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        if viewer_latest_jpeg:
            await asyncio.wait_for(ws.send_bytes(viewer_latest_jpeg), timeout=VIEWER_SEND_TIMEOUT_SEC)
    except Exception:
        pass
    # Keepalive: periodically send the latest frame (if any) so dead sockets are pruned
    # even when the camera stream stalls.
    try:
        while True:
            await asyncio.sleep(15)
            jpeg = viewer_latest_jpeg
            if jpeg:
                try:
                    await asyncio.wait_for(ws.send_bytes(jpeg), timeout=VIEWER_SEND_TIMEOUT_SEC)
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        try: 
            camera_viewers.remove(ws)
        except Exception: 
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)
# === 新增：注册给 bridge_io 的发送回调（把 JPEG 广播给 /ws/viewer） ===
@app.on_event("startup")
async def on_startup_register_bridge_sender():
    # 保存主线程的事件循环
    main_loop = asyncio.get_event_loop()
    
    def _sender(jpeg_bytes: bytes):
        # 注意：这个函数可能在非协程线程里被调用，需要切回主事件循环
        try:
            # 检查事件循环状态，避免在关闭时发送
            if main_loop.is_closed():
                return
            
            # 标记YOLO已经开始发送处理后的帧
            global yolomedia_sending_frames, _yolomedia_last_frame_t
            if not yolomedia_sending_frames:
                yolomedia_sending_frames = True
                print("[YOLOMEDIA] 开始发送处理后的帧，切换到YOLO画面", flush=True)

            # Track latest YOLO frame time for stale fallback logic
            _yolomedia_last_frame_t = time.monotonic()
            
            # 不在这里直接广播（慢client会堆积任务导致卡死）；只更新最新帧
            main_loop.call_soon_threadsafe(_viewer_set_latest, jpeg_bytes)
        except Exception as e:
            # 只在非预期错误时打印日志
            if "Event loop is closed" not in str(e):
                print(f"[DEBUG] _sender error: {e}", flush=True)

    bridge_io.set_sender(_sender)

@app.on_event("startup")
async def on_startup():
    # Start viewer broadcaster (latest-frame only)
    global viewer_new_frame_evt, viewer_broadcast_task
    if viewer_new_frame_evt is None:
        viewer_new_frame_evt = asyncio.Event()
    if viewer_broadcast_task is None or viewer_broadcast_task.done():
        viewer_broadcast_task = asyncio.create_task(_viewer_broadcast_loop())

@app.on_event("shutdown")
async def on_shutdown():
    """应用关闭时的清理工作"""
    print("[SHUTDOWN] 开始清理资源...")

    # Stop viewer broadcaster
    global viewer_broadcast_task
    try:
        if viewer_broadcast_task and not viewer_broadcast_task.done():
            viewer_broadcast_task.cancel()
    except Exception:
        pass
    
    # 停止YOLO媒体处理
    stop_yolomedia()

    print("[SHUTDOWN] 资源清理完成")

# app_main.py —— 在文件里已有的 @app.on_event("startup") 之后，再加一个新的 startup 钩子


# --- 导出接口（可选） ---
def get_last_frames():
    return last_frames

def get_camera_ws():
    return esp32_camera_ws

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8081,
        log_level="warning", access_log=False,
        loop="asyncio", workers=1, reload=False
    )
