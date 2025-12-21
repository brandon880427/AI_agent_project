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
VISION_AGENT_PREPROCESS_JPEG_QUALITY = int(os.getenv("VISION_AGENT_PREPROCESS_JPEG_QUALITY", "90"))


def _preprocess_jpeg(jpeg_bytes: bytes) -> bytes:
    if not VISION_AGENT_PREPROCESS:
        return jpeg_bytes
    try:
        from PIL import Image, ImageEnhance, ImageOps

        im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
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
        await self._emit_text("[AI] （agent）已启动。")
        try:
            while not self._stop_evt.is_set():
                t0 = time.monotonic()
                try:
                    async with self._lock:
                        await self._step_once_locked(user_task=None)
                except Exception as e:
                    self._last_error = str(e)
                    try:
                        await self._emit_text(f"[AI] （agent）运行异常：{e}")
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
                await self._emit_text("[AI] （agent）已停止。")
            except Exception:
                pass

    async def _step_once_locked(self, *, user_task: Optional[str]) -> Dict[str, object]:
        jpeg = self._get_latest_jpeg()
        if not jpeg:
            self._last_error = "no_frame"
            await self._emit_text("[AI] （agent）当前没有相机画面，等待中…")
            return {"ok": False, "error": "no_frame"}

        system_prompt = (
            "你是一个用于盲人辅助导航的视觉Agent。"
            "你会看到来自胸前/眼镜相机的一帧画面。"
            "你的目标：用中文给出非常简短、可执行的提醒（1-2句），"
            "优先关注：障碍物/台阶/车辆/行人/红绿灯/斑马线/门口/转向。"
            "如果画面不确定，请说不确定并给出保守建议。"
            "输出必须是 JSON，字段固定为：summary, hazards, action。"
            "hazards 是字符串数组；action 是一句可执行指令。"
        )

        user_prompt = (user_task or "请分析当前画面并给出提醒。")

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
            "messages": messages,
        }

        timeout = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
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
                "summary": content[:200] if content else "（空）",
                "hazards": [],
                "action": "请放慢脚步，注意周围环境。",
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
        haz_txt = ("；".join(hazards)) if hazards else "无"
        out = f"[AI] （agent）{summary} | 风险: {haz_txt} | 建议: {action}"
        await self._emit_text(out)

        return {"ok": True, "raw": content, "parsed": obj}
