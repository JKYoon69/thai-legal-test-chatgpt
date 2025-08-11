# -*- coding: utf-8 -*-
"""
OpenAI Chat Completions 기반 요약
- call_openai_summary(): 단건
- call_openai_batch(): 배치(여러 섹션 → [[ID]] … 블록으로 회수)
반환값에 usage/ms/model/est_cost_usd 포함(Report에 적재용)
"""
from __future__ import annotations
import os, time, re
from typing import Optional, Dict, Any, List
from openai import OpenAI

# ===== 설정 =====
MODEL_DEFAULT = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4.1-mini")

# 단가(1K tokens, USD) — 필요시 환경변수로 덮어쓰기 가능
PRICE = {
    # 예시 단가(모델별 최신 단가는 대시보드 기준으로 갱신하세요)
    "gpt-4.1-mini": {"in": float(os.getenv("PRICE_GPT41MINI_IN", "0.003")), "out": float(os.getenv("PRICE_GPT41MINI_OUT","0.006"))},
    "gpt-4o-mini": {"in": float(os.getenv("PRICE_GPT4OMINI_IN", "0.0005")), "out": float(os.getenv("PRICE_GPT4OMINI_OUT","0.0015"))},
    "gpt-4o":      {"in": float(os.getenv("PRICE_GPT4O_IN", "0.005")), "out": float(os.getenv("PRICE_GPT4O_OUT","0.015"))},
}

def _get_secret(name: str) -> Optional[str]:
    # Streamlit secrets 우선, 없으면 env
    try:
        import streamlit as st  # type: ignore
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets[name]  # type: ignore
    except Exception:
        pass
    return os.getenv(name)

def _client() -> OpenAI:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAI(api_key=api_key)

# ==================== 단건 ====================
def call_openai_summary(
    text: str, *,
    model: str = MODEL_DEFAULT,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    단문 요약(태국어 불릿 3~6개)
    """
    system = "สรุปข้อความเป็นภาษาไทยแบบสั้น กระชับ เป็นข้อ ๆ (3–6 ข้อ)"
    prompt = text.strip()
    cli = _client()
    t0 = time.time()
    try:
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text_out = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        ms = int((time.time()-t0)*1000)
        est_cost = None
        if usage and model in PRICE:
            est_cost = (usage.prompt_tokens/1000.0)*PRICE[model]["in"] + (usage.completion_tokens/1000.0)*PRICE[model]["out"]
        return {"ok": True, "text": text_out, "usage": dict(usage) if usage else None, "ms": ms, "model": model, "est_cost_usd": est_cost}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# ==================== 배치 ====================

BATCH_INSTRUCTIONS = (
    "ต่อไปนี้คือข้อความกฎหมายหลายส่วน แต่ละส่วนถูกคั่นด้วยเส้น '=== [ID:<ไอดี>] ==='.\n"
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
    sections: [{"id": "A0001", "title": "...", "breadcrumbs":"...", "text":"..."}]
    반환: {"ok":True, "map":{id:text,...}, "raw":full_text, "usage":{...}, "ms":..., "model":..., "est_cost_usd": float|None}
    """
    if not sections:
        return {"ok": True, "map": {}, "raw": "", "usage": {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}, "ms":0, "model": model, "est_cost_usd": 0.0}

    # 프롬프트 구성
    blocks = []
    for s in sections:
        head = f"=== [ID:{s['id']}] ==="
        title = s.get("title","").strip()
        crumbs = s.get("breadcrumbs","").strip()
        header = f"{head}\n{title}\n{crumbs}".strip()
        body = (s.get("text") or "").strip()
        blocks.append(header + "\n\n" + body)
    user_text = BATCH_INSTRUCTIONS + "\n\n" + "\n\n".join(blocks)

    cli = _client()
    t0 = time.time()
    try:
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"คุณเป็นผู้ช่วยสรุปกฎหมายไทยอย่างเคร่งครัดตามรูปแบบที่กำหนด"},
                      {"role":"user","content":user_text}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        out = (resp.choices[0].message.content or "").strip()
        ms = int((time.time()-t0)*1000)
        usage = getattr(resp, "usage", None)
        est_cost = None
        if usage and model in PRICE:
            est_cost = (usage.prompt_tokens/1000.0)*PRICE[model]["in"] + (usage.completion_tokens/1000.0)*PRICE[model]["out"]

        chunks = ID_HEADER_RE.split(out)
        id_map: Dict[str,str] = {}
        if len(chunks) >= 3:
            it = iter(chunks[1:])
            for id_str, block in zip(it, it):
                id_map[id_str.strip()] = block.strip()
        else:
            if sections:
                id_map[sections[0]["id"]] = out.strip()

        u = None
        if usage:
            u = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}

        return {"ok": True, "map": id_map, "raw": out, "usage": u, "ms": ms, "model": model, "est_cost_usd": est_cost}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
