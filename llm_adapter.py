# -*- coding: utf-8 -*-
"""
LLM adapter (OpenAI + Gemini) with robust fallbacks.

- OpenAI 우선: gpt-4.1-mini-2025-04-14 → gpt-4.1-mini → gpt-5
- Gemini 백업: gemini-2.5-flash
- 모든 호출은 실패 사유를 diag["errors"]에 남긴다.
- JSON 스키마는 '검증'만 로컬에서 한다(생성은 모델 프리).
"""
from __future__ import annotations
import os, json, time, sqlite3, threading
from typing import Dict, Any, Optional, List, Tuple

_openai_client = None
_gemini = None

def _sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _load_secret(name: str) -> Optional[str]:
    # streamlit secrets 우선 → env 폴백
    try:
        import streamlit as st  # type: ignore
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

# -------- JSON Schemas -------- #

LAW_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string", "enum": ["act","code","regulation","constitution","unknown"]},
        "law_name": {"type": "string"},
        "year_be":  {"type": ["string","null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "notes": {"type": "string"}
    },
    "required": ["doc_type","law_name","confidence"],
    "additionalProperties": False
}

DESC_SCHEMA = {
    "type": "object",
    "properties": {
        "brief": {"type": "string", "maxLength": 180},
        "topics": {"type": "array","items":{"type":"string"},"minItems":1,"maxItems":6},
        "negations": {"type":"array","items":{"type":"string"},"minItems":0,"maxItems":4},
        "confidence": {"type":"number","minimum":0.0,"maximum":1.0}
    },
    "required": ["brief","topics","confidence"],
    "additionalProperties": False
}

# -------- SQLite cache -------- #

_DB_PATH = os.environ.get("LLM_CACHE_DB", "llm_cache.sqlite3")
_DB_LOCK = threading.Lock()

def _init_db():
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cache(
          provider TEXT, model TEXT, task TEXT, key TEXT,
          value TEXT, created_at REAL,
          PRIMARY KEY(provider,model,task,key)
        )""")
        conn.commit(); conn.close()

def _cache_get(provider, model, task, key):
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        cur = conn.execute(
            "SELECT value FROM cache WHERE provider=? AND model=? AND task=? AND key=?",
            (provider,model,task,key)
        )
        row = cur.fetchone(); conn.close()
        return json.loads(row[0]) if row else None

def _cache_set(provider, model, task, key, value):
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
                     (provider,model,task,key,json.dumps(value,ensure_ascii=False),time.time()))
        conn.commit(); conn.close()

# -------- Providers -------- #

def _ensure_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # type: ignore
        api_key = _load_secret("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def _ensure_gemini():
    global _gemini
    if _gemini is None:
        import google.generativeai as genai  # type: ignore
        api_key = _load_secret("GOOGLE_API_KEY") or _load_secret("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")
        genai.configure(api_key=api_key)
        _gemini = genai
    return _gemini

def _validate_json(obj: Dict[str,Any], schema: Dict[str,Any]) -> bool:
    try:
        import jsonschema  # type: ignore
        jsonschema.validate(instance=obj, schema=schema)
        return True
    except Exception:
        return False

# -------- Router -------- #

class LLMRouter:
    def __init__(self,
                 primary_model: str = "gpt-4.1-mini-2025-04-14",
                 fallback1_model: str = "gemini-2.5-flash",
                 fallback2_model: str = "gpt-5",
                 law_conf_threshold: float = 0.75,
                 desc_conf_threshold: float = 0.60):
        _init_db()
        self.primary_model = primary_model
        self.fallback1_model = fallback1_model
        self.fallback2_model = fallback2_model
        self.law_conf_threshold = law_conf_threshold
        self.desc_conf_threshold = desc_conf_threshold

    def _openai_primary_candidates(self) -> List[str]:
        cands = [self.primary_model]
        if self.primary_model.startswith("gpt-4.1-mini") and self.primary_model != "gpt-4.1-mini":
            cands.append("gpt-4.1-mini")
        return cands

    # ---- (A) doc metadata ----
    def lawname_doctype(self, snippet: str):
        sys = "Answer in Thai where appropriate. Return JSON only."
        prompt = (
            "คุณคือผู้ช่วย metadata สำหรับเอกสารกฎหมาย สกัดหัวเรื่อง/ประเภท แล้วส่งออก JSON ตามสคีมา:\n"
            "- doc_type: act|code|regulation|constitution|unknown\n"
            "- law_name: ชื่อกฎหมายแบบสั้นแต่ครบ\n"
            "- year_be: พ.ศ. ถ้ามี ไม่พบให้ null\n"
            "- confidence: 0..1\n"
            "ห้ามคาดเดาเกินเนื้อหา ตอบ JSON เท่านั้น"
        )
        key = _sha1("LAW|" + snippet[:1600])
        diag = {"route": [], "cached": False, "errors": []}

        # cache
        for provider, model in [("openai", self.primary_model),
                                ("google", self.fallback1_model),
                                ("openai", self.fallback2_model)]:
            cached = _cache_get(provider, model, "law", key)
            if cached and _validate_json(cached, LAW_SCHEMA):
                diag["route"].append({"provider":provider,"model":model,"used":"cache"})
                diag["cached"] = True
                return cached, diag

        # openai primary
        for mdl in self._openai_primary_candidates():
            obj, err = self._call_openai_json(mdl, sys, prompt, snippet, 0)
            if obj and _validate_json(obj, LAW_SCHEMA):
                _cache_set("openai", mdl, "law", key, obj)
                diag["route"].append({"provider":"openai","model":mdl,"used":"live"})
                return obj, diag
            if err: diag["errors"].append({"provider":"openai","model":mdl,"error":err})

        # gemini
        obj, err = self._call_gemini_json(self.fallback1_model, sys, prompt, snippet, 0.0)
        if obj and _validate_json(obj, LAW_SCHEMA):
            _cache_set("google", self.fallback1_model, "law", key, obj)
            diag["route"].append({"provider":"google","model":self.fallback1_model,"used":"live"})
            return obj, diag
        if err: diag["errors"].append({"provider":"google","model":self.fallback1_model,"error":err})

        # gpt-5
        obj, err = self._call_openai_json(self.fallback2_model, sys, prompt, snippet, 0)
        if obj and _validate_json(obj, LAW_SCHEMA):
            _cache_set("openai", self.fallback2_model, "law", key, obj)
            diag["route"].append({"provider":"openai","model":self.fallback2_model,"used":"live"})
            return obj, diag
        if err: diag["errors"].append({"provider":"openai","model":self.fallback2_model,"error":err})

        diag["route"].append({"provider":"all","model":"failed","used":"none"})
        return None, diag

    # ---- (B) chunk descriptors ----
    def describe_chunk(self, chunk_text: str, section_label: str, breadcrumbs: List[str]):
        sys = "Return JSON only. Do not include article numbers in brief."
        prompt = (
            "สรุปเจตนารมณ์ของข้อความกฎหมายแบบย่อ แล้วส่งออก JSON:\n"
            "- brief ≤ 180 (ห้ามใส่เลขมาตรา/ข้อ)\n"
            "- topics 3–6\n"
            "- negations (ไม่/ห้าม/เว้นแต่) 0–4 ถ้ามี\n"
            "ห้ามคาดเดาเกินเนื้อหา"
        )
        snippet = chunk_text[:1600]
        key = _sha1("DESC|" + (section_label or "") + "|" + "|".join(breadcrumbs or []) + "|" + snippet)
        diag = {"route": [], "cached": False, "errors": []}

        for provider, model in [("openai", self.primary_model),
                                ("google", self.fallback1_model),
                                ("openai", self.fallback2_model)]:
            cached = _cache_get(provider, model, "desc", key)
            if cached and _validate_json(cached, DESC_SCHEMA):
                diag["route"].append({"provider":provider,"model":model,"used":"cache"})
                diag["cached"] = True
                return cached, diag

        # openai
        for mdl in self._openai_primary_candidates():
            obj, err = self._call_openai_json(mdl, sys, prompt, snippet, 0.3)
            if obj and _validate_json(obj, DESC_SCHEMA):
                _cache_set("openai", mdl, "desc", key, obj)
                diag["route"].append({"provider":"openai","model":mdl,"used":"live"})
                return obj, diag
            if err: diag["errors"].append({"provider":"openai","model":mdl,"error":err})

        # gemini
        obj, err = self._call_gemini_json(self.fallback1_model, sys, prompt, snippet, 0.3)
        if obj and _validate_json(obj, DESC_SCHEMA):
            _cache_set("google", self.fallback1_model, "desc", key, obj)
            diag["route"].append({"provider":"google","model":self.fallback1_model,"used":"live"})
            return obj, diag
        if err: diag["errors"].append({"provider":"google","model":self.fallback1_model,"error":err})

        # gpt-5
        obj, err = self._call_openai_json(self.fallback2_model, sys, prompt, snippet, 0.3)
        if obj and _validate_json(obj, DESC_SCHEMA):
            _cache_set("openai", self.fallback2_model, "desc", key, obj)
            diag["route"].append({"provider":"openai","model":self.fallback2_model,"used":"live"})
            return obj, diag
        if err: diag["errors"].append({"provider":"openai","model":self.fallback2_model,"error":err})

        diag["route"].append({"provider":"all","model":"failed","used":"none"})
        return None, diag

    def describe_chunks_batch(self, items: List[Tuple[str,str,List[str]]]):
        results, log = [], {"calls":0,"ok":0,"fail":0,"routes":[]}
        import time as _t
        for t, lab, bc in items:
            t0 = _t.time()
            obj, diag = self.describe_chunk(t, lab, bc)
            dt = round(_t.time()-t0, 3)
            log["calls"] += 1; log["routes"].append({"diag":diag,"latency_s":dt})
            if obj: results.append(obj); log["ok"] += 1
            else: results.append(None); log["fail"] += 1
        return results, log

    # -------- provider calls: return (obj, error_str) -------- #

    def _call_openai_json(self, model, system, prompt, text, temperature):
        """
        SDK 호환 계층:
        1) client.responses.create(..., response_format=...) → 실패 시
        2) client.responses.create(... )                     → 실패 시
        3) client.chat.completions.create(..., response_format={"type":"json_object"})
        """
        try:
            client = _ensure_openai()
            msg = f"{prompt}\n\n<document>\n{text}\n</document>"
            # 1) Responses API + response_format
            try:
                rf = {"type":"json_schema","json_schema":{"name":"Schema","schema":{"type":"object"},"strict":False}}
                resp = client.responses.create(
                    model=model,
                    input=[{"role":"system","content":system},{"role":"user","content":msg}],
                    temperature=temperature,
                    response_format=rf,
                    max_output_tokens=512,
                )
                content = None
                if hasattr(resp,"output") and resp.output and getattr(resp.output[0],"content",None):
                    content = resp.output[0].content[0].text
                if not content and hasattr(resp,"choices"):
                    content = resp.choices[0].message["content"]
                if content: return (json.loads(content), None)
            except TypeError:
                # 2) Responses API (no response_format)
                try:
                    resp = client.responses.create(
                        model=model,
                        input=[{"role":"system","content":system},{"role":"user","content":msg}],
                        temperature=temperature,
                        max_output_tokens=512,
                    )
                    content = None
                    if hasattr(resp,"output") and resp.output and getattr(resp.output[0],"content",None):
                        content = resp.output[0].content[0].text
                    if not content and hasattr(resp,"choices"):
                        content = resp.choices[0].message["content"]
                    if content: return (json.loads(content), None)
                except Exception as e2:
                    # 3) Chat Completions JSON 모드
                    try:
                        cc = client.chat.completions.create(
                            model=model,
                            messages=[{"role":"system","content":system},{"role":"user","content":msg}],
                            temperature=temperature,
                            response_format={"type":"json_object"},
                            max_tokens=512
                        )
                        content = cc.choices[0].message.content
                        if content: return (json.loads(content), None)
                    except Exception as e3:
                        return (None, f"{type(e3).__name__}: {str(e3)}"[:180])
            return (None, "Empty response")
        except Exception as e:
            return (None, f"{type(e).__name__}: {str(e)}"[:180])

    def _call_gemini_json(self, model, system, prompt, text, temperature):
        """
        Gemini: schema 파라미터 없이 JSON MIME만 요청(버전간 호환).
        """
        try:
            genai = _ensure_gemini()
            cfg = {"temperature":temperature, "response_mime_type":"application/json", "max_output_tokens":512}
            m = genai.GenerativeModel(model_name=model, system_instruction=system, generation_config=cfg)
            msg = f"{prompt}\n\n<document>\n{text}\n</document>"
            resp = m.generate_content([msg])
            return (json.loads(resp.text) if getattr(resp,"text",None) else None, None)
        except Exception as e:
            return (None, f"{type(e).__name__}: {str(e)}"[:180])
