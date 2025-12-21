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


_MODEL_CACHE: Optional[List[str]] = None


async def _list_ollama_models(client: httpx.AsyncClient) -> List[str]:
    """Return available Ollama model names (best-effort)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    try:
        r = await client.get("/api/tags")
        r.raise_for_status()
        j = r.json()
        models: List[str] = []
        for m in (j.get("models") or []):
            if isinstance(m, dict) and m.get("name"):
                models.append(str(m["name"]))
        _MODEL_CACHE = models
        return models
    except Exception:
        _MODEL_CACHE = []
        return []


async def _choose_model(client: httpx.AsyncClient, requested: str) -> Tuple[Optional[str], Optional[str]]:
    """Pick a usable model; returns (model_name, warning_message)."""
    requested = (requested or "").strip()
    models = await _list_ollama_models(client)
    if not models:
        # Can't reach Ollama or it has no models.
        if requested:
            return None, f"[OLLAMA] No models available (requested '{requested}'). Install a model or set OLLAMA_MODEL."
        return None, "[OLLAMA] No models available. Install a model or set OLLAMA_MODEL."

    if not requested:
        return models[0], f"[OLLAMA] OLLAMA_MODEL not set; using '{models[0]}'."

    # Exact match.
    if requested in models:
        return requested, None
    # Common case: user sets base name (e.g. llama3) while tags include version.
    prefix = requested + ":"
    for m in models:
        if m.startswith(prefix):
            return m, f"[OLLAMA] Requested model '{requested}' not found; using '{m}'."

    return models[0], f"[OLLAMA] Requested model '{requested}' not found; using '{models[0]}'."


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
        chosen_model, warn = await _choose_model(client, OLLAMA_MODEL)
        if warn:
            # Emit warning but continue if we have a model.
            yield OmniStreamPiece(text_delta=warn)
        if not chosen_model:
            return
        try:
            resp = await client.post("/api/generate", json={
                "model": chosen_model,
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

        # Ollama may return error JSON like: {"error": "model 'xxx' not found"}
        if isinstance(j, dict) and j.get("error"):
            err = str(j.get("error"))
            # If model missing, refresh cache once and retry with fallback.
            if "not found" in err and "model" in err:
                global _MODEL_CACHE
                _MODEL_CACHE = None
                chosen_model2, warn2 = await _choose_model(client, OLLAMA_MODEL)
                if warn2:
                    yield OmniStreamPiece(text_delta=warn2)
                if chosen_model2 and chosen_model2 != chosen_model:
                    try:
                        resp2 = await client.post("/api/generate", json={
                            "model": chosen_model2,
                            "prompt": prompt,
                        })
                        try:
                            j2 = resp2.json()
                        except Exception:
                            j2 = None
                        if isinstance(j2, dict) and j2.get("error"):
                            yield OmniStreamPiece(text_delta=f"[OLLAMA ERROR] {j2.get('error')}")
                            return
                        j = j2
                        resp = resp2
                        chosen_model = chosen_model2
                    except Exception as e:
                        yield OmniStreamPiece(text_delta=f"[OLLAMA ERROR] {e}")
                        return
                else:
                    yield OmniStreamPiece(text_delta=f"[OLLAMA ERROR] {err}")
                    return
            else:
                yield OmniStreamPiece(text_delta=f"[OLLAMA ERROR] {err}")
                return

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
            yield OmniStreamPiece(text_delta="[OLLAMA] No response or parse failed")
            return

        # For now Ollama produces text; audio generation/tts is out-of-scope here.
        yield OmniStreamPiece(text_delta=text_output, audio_b64=None)
