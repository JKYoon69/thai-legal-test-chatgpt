# -*- coding: utf-8 -*-
"""
LLM 통합 클라이언트:
- OpenAI Responses API (gpt-4.1-mini-2025-04-14, gpt-5)
- Google Gemini (gemini-2.5-flash, google-genai SDK)
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

def openai_responses_call(
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
        "max_output_tokens": max_output_tokens,
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

    text = getattr(resp, "output_text", None)
    if not text:
        # 중간 버전 폴백
        try:
            text = "\n".join([b.content[0].text for b in getattr(resp, "output", []) if getattr(b, "content", None)])
        except Exception:
            text = None

    if not text:
        return {"ok": False, "error": "openai:empty_output_text", "raw": resp}

    return {"ok": True, "text": text, "raw": resp}


def call_openai_41mini(prompt: str, *, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return openai_responses_call(
        model="gpt-4.1-mini-2025-04-14",
        prompt=prompt,
        instructions="Return concise Thai legal metadata/summary as requested.",
        max_output_tokens=512,
        temperature=0.2,
        json_schema=schema,
    )

def call_openai_gpt5(prompt: str) -> Dict[str, Any]:
    # temperature 금지, max_output_tokens만 사용
    return openai_responses_call(
        model="gpt-5",
        prompt=prompt,
        instructions="Return concise Thai legal metadata/summary as requested.",
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
    if _gemini is None:
        return {"ok": False, "error": "gemini:client_not_initialized"}

    try:
        resp = _gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"max_output_tokens": 512},
        )
        # finish_reason이 STOP(=1)일 때만 quick accessor 신뢰
        cand = resp.candidates[0] if resp.candidates else None
        finish = getattr(cand, "finish_reason", None)
        text = getattr(resp, "text", None) if finish == 1 else None

        if not text:
            # dict로 펼쳐서 Part에서 텍스트 추출
            data = resp.to_dict()
            try:
                text = data["candidates"][0]["content"]["parts"][0].get("text", "")
            except Exception:
                text = ""

        if not text:
            return {"ok": False, "error": f"gemini:no_text (finish_reason={finish})"}

        return {"ok": True, "text": text, "raw": None}
    except Exception as e:
        return {"ok": False, "error": f"gemini:{type(e).__name__}:{e}"}
