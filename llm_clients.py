# -*- coding: utf-8 -*-
"""
LLM 호출: OpenAI Chat Completions (gpt-4.1-mini-2025-04-14)
- JSON 강제 사용 안 함. 프리포맷 요약만 받고, 구조화는 로컬 규칙으로.
- 키는 Streamlit Secrets 또는 환경변수 OPENAI_API_KEY.
"""
from __future__ import annotations
import os, time
from typing import Optional, Dict, Any

def _get_secret(name: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

from openai import OpenAI
_client: Optional[OpenAI] = None

def _client_openai() -> OpenAI:
    global _client
    if _client is None:
        key = _get_secret("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in Secrets or env.")
        _client = OpenAI(api_key=key)
    return _client

def call_openai_summary(
    prompt: str,
    *,
    model: str = "gpt-4.1-mini-2025-04-14",
    max_tokens: int = 512,
    temperature: float = 0.2,
    system: str = (
        "You are a concise Thai legal summarizer. "
        "Always write in Thai. Avoid prefaces, disclaimers, citations, and markdown headings."
    ),
) -> Dict[str, Any]:
    """
    Chat Completions 단일 호출. 성공 시 {'ok':True,'text':...,'ms':...} 반환.
    실패 시 {'ok':False,'error':...,'ms':...}.
    """
    t0 = time.time()
    try:
        cli = _client_openai()
        resp = cli.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "text": text, "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "ms": int((time.time() - t0) * 1000)}
