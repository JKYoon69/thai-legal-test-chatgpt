# -*- coding: utf-8 -*-
"""
LLM 호출 (OpenAI Chat Completions)
- 단일 요약 call_openai_summary()
- 배치 요약 call_openai_batch() : 여러 섹션을 한 번에 요청 → [[ID]] 블록으로 분리해서 반환
"""

from __future__ import annotations
import os, time, re
from typing import Optional, Dict, Any, List
from openai import OpenAI

def _get_secret(name: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

_client: Optional[OpenAI] = None

def _client_openai() -> OpenAI:
    global _client
    if _client is None:
        key = _get_secret("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=key)
    return _client


MODEL_DEFAULT = "gpt-4.1-mini-2025-04-14"
SYSTEM_SUMMARY = (
    "You are a concise Thai legal summarizer. "
    "Always write in Thai. Avoid prefaces, disclaimers, citations, and markdown headings."
)

def call_openai_summary(
    prompt: str,
    *,
    model: str = MODEL_DEFAULT,
    max_tokens: int = 512,
    temperature: float = 0.2,
    system: str = SYSTEM_SUMMARY,
) -> Dict[str, Any]:
    t0 = time.time()
    try:
        cli = _client_openai()
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "text": text, "ms": int((time.time()-t0)*1000)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "ms": int((time.time()-t0)*1000)}


# -------------------------- 배치 요약 --------------------------

BATCH_INSTRUCTIONS = (
    "ต่อไปนี้คือข้อความกฎหมายหลายส่วน แต่ละส่วนถูกคั่นด้วยเส้น '=== [ID:<เลขไอดี>] ==='.\n"
    "โปรดสรุปแต่ละส่วนแบบสั้น กระชับ เป็นข้อ ๆ (3–6 ข้อ) เป็นภาษาไทยเท่านั้น\n"
    "รูปแบบคำตอบ **ต้องเป็นดังนี้เท่านั้น** (ห้ามมีหัวเรื่อง/คำนำอื่น):\n"
    "[[<ID>]]\n"
    "- ข้อ 1 …\n"
    "- ข้อ 2 …\n"
    "(ขึ้นบรรทัดว่าง 1 บรรทัดก่อนขึ้น ID ถัดไป)\n"
)

ID_HEADER_RE = re.compile(r"^\s*\[\[(?P<id>[^\]]+)\]\]\s*$", re.MULTILINE)

def call_openai_batch(
    sections: List[Dict[str, str]],
    *,
    model: str = MODEL_DEFAULT,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    sections: [{id,title,breadcrumbs,text}]
    반환: {"ok":True, "map":{id:str_summary}, "raw":full_text} 또는 {"ok":False,"error":...}
    """
    # 프롬프트 구성
    parts = [BATCH_INSTRUCTIONS]
    for s in sections:
        parts.append(f"=== [ID:{s['id']}] ===")
        if s.get("title"):
            parts.append(f"<section>{s['title']}</section>")
        if s.get("breadcrumbs"):
            parts.append(f"<breadcrumbs>{s['breadcrumbs']}</breadcrumbs>")
        # 입력 길이 안전 한도
        txt = s["text"][:2000]
        parts.append(f"<document>\n{txt}\n</document>\n")
    prompt = "\n".join(parts).strip()

    r = call_openai_summary(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
    if not r.get("ok"):
        return r

    out = r["text"]
    # [[ID]] 블록 파싱
    # [[A1]]\n- ...\n- ...\n\n[[A2]]\n...
    chunks = ID_HEADER_RE.split(out)
    # split 결과: ["(프리루드)", "ID1", "블록1", "ID2", "블록2", ...]
    id_map: Dict[str,str] = {}
    if len(chunks) >= 3:
        # 처음 토큰은 프리루드(가능하면 무시)
        it = iter(chunks[1:])
        for id_str, block in zip(it, it):
            id_map[id_str.strip()] = block.strip()
    else:
        # 모델이 형식을 지키지 않은 경우: 전체를 첫 ID에 몰아넣는 안전 폴백
        if sections:
            id_map[sections[0]["id"]] = out.strip()

    return {"ok": True, "map": id_map, "raw": out}
