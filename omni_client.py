# omni_client.py
# -*- coding: utf-8 -*-
import os, base64, asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple

import httpx

# ===== Ollama client (local LLM) =====
# Use OLLAMA_URL and OLLAMA_MODEL environment variables. If you previously used
# DASHSCOPE_API_KEY for DashScope, it's left untouched for ASR compatibility.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
# Optional API key (for Ollama Cloud or secured endpoints)
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")


class OmniStreamPiece:
    """对外的统一增量数据：text/audio 二选一或同时。"""
    def __init__(self, text_delta: Optional[str] = None, audio_b64: Optional[str] = None):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64

async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = "Cherry",
    audio_format: str = "wav",
) -> AsyncGenerator[OmniStreamPiece, None]:
    """
    发起一轮 Omni-Turbo ChatCompletions 流式对话：
    - content_list: OpenAI chat 的 content，多模态（image_url/text）
    - 以 stream=True 返回
    - 增量产出：OmniStreamPiece(text_delta=?, audio_b64=?)
    """
    # Convert content_list to a plain-text prompt for Ollama
    def _content_list_to_prompt(contents: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in contents:
            if isinstance(item, dict):
                role = item.get("role", "user")
                c = item.get("content", "")
                # if content is list/dict, stringify safely
                if isinstance(c, (list, dict)):
                    try:
                        c = str(c)
                    except Exception:
                        c = ""
                parts.append(f"{role}: {c}")
            else:
                parts.append(str(item))
        return "\n".join(parts)

    prompt = _content_list_to_prompt(content_list)

    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=30.0, headers=headers) as client:
        try:
            resp = await client.post("/api/generate", json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
            })
        except Exception as e:
            # network / connection error
            yield OmniStreamPiece(text_delta=f"[OLLAMA ERROR] {e}")
            return

        text_output: Optional[str] = None
        try:
            j = resp.json()
        except Exception:
            j = None

        def _extract_text(obj: Any) -> Optional[str]:
            # Try common shapes returned by Ollama
            try:
                if not obj:
                    return None
                if isinstance(obj, dict):
                    # results -> content -> text
                    if "results" in obj:
                        texts: List[str] = []
                        for r in obj.get("results", []):
                            content = r.get("content") or r.get("output") or []
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict) and "text" in c:
                                        texts.append(c.get("text") or "")
                                    elif isinstance(c, str):
                                        texts.append(c)
                            elif isinstance(content, str):
                                texts.append(content)
                        if texts:
                            return "\n".join(t for t in texts if t)
                    if "choices" in obj:
                        for c in obj.get("choices", []):
                            if isinstance(c, dict):
                                if "message" in c and isinstance(c["message"], dict):
                                    cont = c["message"].get("content")
                                    if isinstance(cont, str):
                                        return cont
                                if "text" in c and isinstance(c["text"], str):
                                    return c["text"]
                    if "text" in obj and isinstance(obj["text"], str):
                        return obj["text"]
                if isinstance(obj, list):
                    # join any strings found
                    parts: List[str] = []
                    for it in obj:
                        if isinstance(it, str):
                            parts.append(it)
                        elif isinstance(it, dict):
                            t = _extract_text(it)
                            if t:
                                parts.append(t)
                    if parts:
                        return "\n".join(parts)
            except Exception:
                return None
            return None

        text_output = _extract_text(j) or (resp.text if resp.text else None)

        if not text_output:
            yield OmniStreamPiece(text_delta="[OLLAMA] 无响应或解析失败")
            return

        # For now Ollama produces text; audio generation/tts is out-of-scope here.
        yield OmniStreamPiece(text_delta=text_output, audio_b64=None)
