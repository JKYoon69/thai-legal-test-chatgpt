# -*- coding: utf-8 -*-
"""
LLM 통합 클라이언트:
- OpenAI Responses API (gpt-4.1-mini-2025-04-14, gpt-5)
- Google Gemini (gemini-2.5-flash; google-genai SDK)
- Streamlit Secrets 우선, 없으면 환경변수 사용
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

def _get_secret(name: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

# ---------- OpenAI (Responses API) ----------
from openai import OpenAI

_openai = None
def _openai_client() -> OpenAI:
    global _openai
    if _openai is None:
        key = _get_secret("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in Secrets")
        _openai = OpenAI(api_key=key)
    return _openai

def _openai_responses(
    model: str,
    prompt: str,
    *,
    instructions: Optional[str] = None,
    max_output_tokens: int = 512,
    temperature: Optional[float] = None,
    json_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """OpenAI Responses API 통합 호출 (구조적 출력은 가능하면 사용, 실패 시 자동 폴백)"""
    client = _openai_client()
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,  # Responses API는 max_output_tokens 사용
    }
    if instructions:
        kwargs["instructions"] = instructions

    # GPT-5는 temperature 미지원 → 전달하지 않음
    if (temperature is not None) and (not model.startswith("gpt-5")):
        kwargs["temperature"] = temperature

    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": json_schema.get("name", "schema"),
                            "schema": json_schema["schema"],
                            "strict": True},
        }

    try:
        resp = client.responses.create(**kwargs)
    except TypeError as te:
        # 일부 SDK/모델에서 response_format 미지원 → 스키마 강제 제거 폴백
        if "response_format" in str(te) and json_schema:
            kwargs.pop("response_format", None)
            resp = client.responses.create(**kwargs)
        else:
            return {"ok": False, "error": f"openai:{model}:TypeError:{te}"}
    except Exception as e:
        return {"ok": False, "error": f"openai:{model}:{type(e).__name__}:{e}"}

    # 공식 SDK 헬퍼 우선
    text = getattr(resp, "output_text", None)
    if not text:
        # 안전 폴백: 하위 content에서 텍스트 모으기
        pieces = []
        try:
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "content", None):
                    for part in item.content:
                        if getattr(part, "text", None):
                            pieces.append(part.text)
        except Exception:
            pass
        text = "\n".join(pieces) if pieces else None

    if not text:
        return {"ok": False, "error": "openai:empty_output_text", "raw": None}

    return {"ok": True, "text": text, "raw": None}


def call_openai_41mini(prompt: str, *, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _openai_responses(
        model="gpt-4.1-mini-2025-04-14",
        prompt=prompt,
        instructions="Return concise Thai legal metadata/summary as requested (JSON only when asked).",
        max_output_tokens=512,
        temperature=0.2,
        json_schema=schema,
    )

def call_openai_gpt5(prompt: str) -> Dict[str, Any]:
    # temperature 금지, max_output_tokens만 사용
    return _openai_responses(
        model="gpt-5",
        prompt=prompt,
        instructions="Return concise Thai legal metadata/summary as requested (JSON only when asked).",
        max_output_tokens=512,
        temperature=None,
        json_schema=None,
    )

# ---------- Gemini (google-genai) ----------
try:
    from google import genai
    _gemini = genai.Client(api_key=_get_secret("GEMINI_API_KEY"))
except Exception:
    _gemini = None

def call_gemini_flash(prompt: str) -> Dict[str, Any]:
    """Gemini 2.5 Flash 안전 호출. response.text → parts → model_dump 순서로 텍스트 확보."""
    if _gemini is None:
        return {"ok": False, "error": "gemini:client_not_initialized"}

    try:
        resp = _gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"max_output_tokens": 512},
        )

        # 1) quick accessor
        if getattr(resp, "text", None):
            return {"ok": True, "text": resp.text, "raw": None}

        # 2) candidates/parts
        try:
            cand = resp.candidates[0] if resp.candidates else None
            parts = cand.content.parts if cand and cand.content and cand.content.parts else []
            for p in parts:
                if getattr(p, "text", None):
                    return {"ok": True, "text": p.text, "raw": None}
        except Exception:
            pass

        # 3) 전체 직렬화 후 텍스트 키 탐색 (Pydantic model → dict)
        try:
            data = resp.model_dump()
            # 매우 보수적 폴백
            txt = (
                data.get("text")
                or ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts", [{}])[0].get("text")
                or ""
            )
            if txt:
                return {"ok": True, "text": txt, "raw": None}
        except Exception:
            pass

        return {"ok": False, "error": "gemini:no_text"}
    except Exception as e:
        return {"ok": False, "error": f"gemini:{type(e).__name__}:{e}"}
