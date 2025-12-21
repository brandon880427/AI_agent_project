# vision_agent.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import io
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Dict, List, Optional, Tuple

import httpx


VISION_AGENT_PREPROCESS = os.getenv("VISION_AGENT_PREPROCESS", "1") != "0"
VISION_AGENT_PREPROCESS_BRIGHTNESS = float(os.getenv("VISION_AGENT_PREPROCESS_BRIGHTNESS", "1.20"))
VISION_AGENT_PREPROCESS_CONTRAST = float(os.getenv("VISION_AGENT_PREPROCESS_CONTRAST", "1.15"))
VISION_AGENT_PREPROCESS_JPEG_QUALITY = int(os.getenv("VISION_AGENT_PREPROCESS_JPEG_QUALITY", "80"))
VISION_AGENT_MAX_SIDE = int(os.getenv("VISION_AGENT_MAX_SIDE", "512"))
VISION_AGENT_OLLAMA_READ_TIMEOUT_SEC = float(os.getenv("VISION_AGENT_OLLAMA_READ_TIMEOUT_SEC", "120"))
VISION_AGENT_OLLAMA_NUM_PREDICT = int(os.getenv("VISION_AGENT_OLLAMA_NUM_PREDICT", os.getenv("OLLAMA_NUM_PREDICT", "120")))
VISION_AGENT_OLLAMA_TEMPERATURE = float(os.getenv("VISION_AGENT_OLLAMA_TEMPERATURE", os.getenv("OLLAMA_TEMPERATURE", "0.2")))


def _preprocess_jpeg(jpeg_bytes: bytes) -> bytes:
    if not VISION_AGENT_PREPROCESS:
        return jpeg_bytes
    try:
        from PIL import Image, ImageEnhance, ImageOps

        im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        if VISION_AGENT_MAX_SIDE and VISION_AGENT_MAX_SIDE > 0:
            w, h = im.size
            m = max(w, h)
            if m > VISION_AGENT_MAX_SIDE:
                scale = VISION_AGENT_MAX_SIDE / float(m)
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                im = im.resize((nw, nh), Image.BILINEAR)
        im = ImageOps.autocontrast(im)
        if abs(VISION_AGENT_PREPROCESS_BRIGHTNESS - 1.0) > 1e-3:
            im = ImageEnhance.Brightness(im).enhance(VISION_AGENT_PREPROCESS_BRIGHTNESS)
        if abs(VISION_AGENT_PREPROCESS_CONTRAST - 1.0) > 1e-3:
            im = ImageEnhance.Contrast(im).enhance(VISION_AGENT_PREPROCESS_CONTRAST)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=VISION_AGENT_PREPROCESS_JPEG_QUALITY, optimize=True)
        return out.getvalue()
    except Exception:
        return jpeg_bytes


@dataclass
class AgentStatus:
    running: bool
    interval_sec: float
    last_step_at: float
    steps: int
    last_error: str


class VisionAgent:
    def __init__(
        self,
        *,
        ollama_url: str,
        ollama_model: str,
        get_latest_jpeg: Callable[[], Optional[bytes]],
        emit_text: Callable[[str], Awaitable[None]],
        interval_sec: float = 2.5,
        history_max: int = 12,
    ) -> None:
        self._ollama_url = (ollama_url or "").rstrip("/")
        self._ollama_model = (ollama_model or "").strip()
        self._get_latest_jpeg = get_latest_jpeg
        self._emit_text = emit_text
        self._interval_sec = float(interval_sec)
        self._history: Deque[Dict[str, str]] = deque(maxlen=int(history_max))

        self._task: Optional[asyncio.Task] = None
        self._stop_evt = asyncio.Event()
        self._lock = asyncio.Lock()

        self._steps = 0
        self._last_step_at = 0.0
        self._last_error = ""

    def status(self) -> AgentStatus:
        return AgentStatus(
            running=self.running,
            interval_sec=self._interval_sec,
            last_step_at=self._last_step_at,
            steps=self._steps,
            last_error=self._last_error,
        )

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def configure(self, *, interval_sec: Optional[float] = None) -> None:
        if interval_sec is not None:
            self._interval_sec = max(0.2, float(interval_sec))

    async def start(self) -> None:
        if self.running:
            return
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if not self.running:
            return
        self._stop_evt.set()
        task = self._task
        if task:
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except Exception:
                pass
        self._task = None

    async def step_once(self, *, user_task: Optional[str] = None) -> Dict[str, object]:
        async with self._lock:
            return await self._step_once_locked(user_task=user_task)

    async def _run_loop(self) -> None:
        await self._emit_text("[AI] (agent) Started.")
        try:
            while not self._stop_evt.is_set():
                t0 = time.monotonic()
                try:
                    async with self._lock:
                        await self._step_once_locked(user_task=None)
                except Exception as e:
                    self._last_error = str(e).strip() or type(e).__name__
                    try:
                        if isinstance(e, httpx.ReadTimeout):
                            msg = (
                                "Timed out waiting for Ollama. Try smaller frames (VISION_AGENT_MAX_SIDE=512) or increase VISION_AGENT_OLLAMA_READ_TIMEOUT_SEC."
                            )
                        else:
                            msg = str(e).strip() or type(e).__name__
                        await self._emit_text(f"[AI] (agent) Error: {msg}")
                    except Exception:
                        pass

                dt = time.monotonic() - t0
                sleep_s = max(0.0, self._interval_sec - dt)
                try:
                    await asyncio.wait_for(self._stop_evt.wait(), timeout=sleep_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            try:
                await self._emit_text("[AI] (agent) Stopped.")
            except Exception:
                pass

    async def _step_once_locked(self, *, user_task: Optional[str]) -> Dict[str, object]:
        jpeg = self._get_latest_jpeg()
        if not jpeg:
            self._last_error = "no_frame"
            await self._emit_text("[AI] (agent) No camera frame yet; waiting...")
            return {"ok": False, "error": "no_frame"}

        system_prompt = (
            "You are a vision agent for blind-assistance navigation. "
            "You will see one frame from a chest/glasses camera. "
            "Goal: provide very short, actionable guidance in ENGLISH (1-2 sentences). "
            "Prioritize: obstacles/steps/vehicles/pedestrians/traffic lights/crosswalks/doors/turning. "
            "If uncertain, say so and give conservative advice. "
            "Output MUST be JSON only with fixed fields: summary, hazards, action. "
            "hazards is an array of short strings; action is one executable instruction."
        )

        user_prompt = (user_task or "Analyze the current frame and give guidance.")

        jpeg = _preprocess_jpeg(jpeg)
        img_b64 = base64.b64encode(jpeg).decode("ascii")
        messages: List[Dict[str, object]] = [
            {"role": "system", "content": system_prompt},
        ]
        for m in list(self._history):
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_prompt, "images": [img_b64]})

        payload = {
            "model": self._ollama_model,
            "stream": False,
            "options": {
                "num_predict": VISION_AGENT_OLLAMA_NUM_PREDICT,
                "temperature": VISION_AGENT_OLLAMA_TEMPERATURE,
            },
            "messages": messages,
        }

        timeout = httpx.Timeout(connect=5.0, read=VISION_AGENT_OLLAMA_READ_TIMEOUT_SEC, write=30.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{self._ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()

        content = ""
        try:
            content = (data.get("message") or {}).get("content") or ""
        except Exception:
            content = ""
        content = (content or "").strip()

        obj: Dict[str, object]
        try:
            # Some models wrap JSON in ```json ... ``` fences or add pre/post text.
            s = content
            l = s.find("{")
            r = s.rfind("}")
            if l != -1 and r != -1 and r > l:
                s = s[l : r + 1]
            obj = json.loads(s)
        except Exception:
            # Fallback: wrap plain text
            obj = {
                "summary": content[:200] if content else "(empty)",
                "hazards": [],
                "action": "Slow down and watch your surroundings.",
            }

        summary = str(obj.get("summary") or "").strip()
        hazards = obj.get("hazards")
        if not isinstance(hazards, list):
            hazards = []
        hazards = [str(x)[:60] for x in hazards if str(x).strip()]
        action = str(obj.get("action") or "").strip()

        self._history.append({"role": "assistant", "content": content})

        self._steps += 1
        self._last_step_at = time.time()
        self._last_error = ""

        # Emit as one concise chat line.
        haz_txt = ("; ".join(hazards)) if hazards else "none"
        out = f"[AI] (agent) {summary} | hazards: {haz_txt} | action: {action}"
        await self._emit_text(out)

        return {"ok": True, "raw": content, "parsed": obj}
