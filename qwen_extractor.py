# qwen_extractor.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple
import os

try:
    from openai import OpenAI
except Exception:  # openai is optional in LITE mode
    OpenAI = None  # type: ignore[assignment]

# —— 本地优先映射（可随时扩充/改名）——
LOCAL_CN2EN = {
    "红牛": "Red_Bull",
    "ad钙奶": "AD_milk",
    "ad 钙奶": "AD_milk",
    "ad": "AD_milk",
    "钙奶": "AD_milk",
    "矿泉水": "bottle",
    "水瓶": "bottle",
    "可乐": "coke",
    "雪碧": "sprite",
}

def _make_client() -> OpenAI:
    # 复用你百炼兼容端点；支持从环境变量读取
    base_url = os.getenv("DASHSCOPE_COMPAT_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    if OpenAI is None:
        raise RuntimeError("Optional dependency missing: pip install openai")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY/QWEN_API_KEY/OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url)

PROMPT_SYS = (
    "You are a label normalizer. Convert the given Chinese object "
    "description into a short, lowercase English YOLO/vision class name "
    "(1~3 words). If multiple are given, return the single most likely one. "
    "Output ONLY the label, no punctuation."
)

def extract_english_label(query_cn: str) -> Tuple[str, str]:
    """
    返回 (label_en, source)；source ∈ {'local', 'qwen', 'fallback'}
    """
    q = (query_cn or "").strip().lower()
    if q in LOCAL_CN2EN:
        return LOCAL_CN2EN[q], "local"

    # 简单规则：去掉前缀修饰词
    for k, v in LOCAL_CN2EN.items():
        if k in q:
            return v, "local"

    # 调用 Qwen Turbo（兼容 Chat Completions）
    try:
        client = _make_client()
        msgs = [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user",   "content": query_cn.strip()},
        ]
        rsp = client.chat.completions.create(
            model=os.getenv("QWEN_MODEL", "qwen-turbo"),
            messages=msgs,
            stream=False
        )
        label = (rsp.choices[0].message.content or "").strip()
        # 清洗一下
        label = label.replace(".", "").replace(",", "").replace("  ", " ").strip()
        # 兜底：空就回 'bottle'
        return (label or "bottle"), "qwen"
    except Exception:
        return "bottle", "fallback"
