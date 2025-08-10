# -*- coding: utf-8 -*-
"""
LLM adapter for:
  (A) law_name / doc_type correction (fallback only)
  (B) per-chunk descriptors (brief/topics/negations)

Routing:
  1) OpenAI gpt-4.1-mini (primary)
  2) Google Gemini 2.5 Flash (fallback-1)
  3) OpenAI gpt-5 (fallback-2)

- Strict JSON schema validation
- Temperature conservative
- On-disk cache (sqlite) by (provider, model, task, key) to minimize cost

API keys:
  - OPENAI_API_KEY
  - GOOGLE_API_KEY
Also supports Streamlit secrets if available (st.secrets["OPENAI_API_KEY"] / ["GOOGLE_API_KEY"])
"""
from __future__ import annotations
import os, json, hashlib, time, sqlite3, threading
from typing import Dict, Any, Optional, List, Tuple

# Optional import guards: we import lazily
_openai_client = None
_gemini = None

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _load_secret(name: str) -> Optional[str]:
    # Streamlit secrets compatible; silent fallback to env
    try:
        import streamlit as st  # type: ignore
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

# ────────── JSON Schema (common) ────────── #

LAW_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string", "enum": ["act","code","regulation","constitution","unknown"]},
        "law_name": {"type": "string"},
        "year_be":  {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "notes": {"type": "string"}
    },
    "required": ["doc_type","law_name","confidence"],
    "additionalProperties": False,
}

DESC_SCHEMA = {
    "type": "object",
    "properties": {
        "brief": {"type": "string", "maxLength": 180},   # 140~180자 요약
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1, "maxItems": 6
        },
        "negations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0, "maxItems": 4
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["brief","topics","confidence"],
    "additionalProperties": False,
}

# ────────── SQLite cache ────────── #

_DB_PATH = os.environ.get("LLM_CACHE_DB", "llm_cache.sqlite3")
_DB_LOCK = threading.Lock()

def _init_db():
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        try:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                task TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (provider, model, task, key)
            );
            """)
            conn.commit()
        finally:
            conn.close()

def _cache_get(provider: str, model: str, task: str, key: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        try:
            cur = conn.execute(
                "SELECT value FROM cache WHERE provider=? AND model=? AND task=? AND key=?",
                (provider, model, task, key)
            )
            row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

def _cache_set(provider: str, model: str, task: str, key: str, value: Dict[str, Any]) -> None:
    with _DB_LOCK:
        conn = sqlite3.connect(_DB_PATH)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache(provider, model, task, key, value, created_at) VALUES (?,?,?,?,?,?)",
                (provider, model, task, key, json.dumps(value, ensure_ascii=False), time.time())
            )
            conn.commit()
        finally:
            conn.close()

# ────────── Providers ────────── #

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
        api_key = _load_secret("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        _gemini = genai
    return _gemini

# ────────── Validator ────────── #

def _validate_json(obj: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    try:
        import jsonschema  # type: ignore
        jsonschema.validate(instance=obj, schema=schema)
        return True
    except Exception:
        return False

# ────────── Router ────────── #

class LLMRouter:
    def __init__(self,
                 primary_model: str = "gpt-4.1-mini",
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

    # ------------- (A) law_name / doc_type ------------- #

    def lawname_doctype(self, snippet: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Returns (result, diag). result is dict with LAW_SCHEMA or None if failed.
        """
        prompt = (
            "You are a legal metadata assistant. Read the Thai law header snippet and output JSON only.\n"
            "- Determine doc_type from: act, code, regulation, constitution, unknown\n"
            "- Extract law_name (short but complete title), and year in Buddhist Era (พ.ศ.) if present.\n"
            "- Do not invent; if uncertain, set year_be to null and lower confidence.\n"
            "Output only JSON.\n"
        )
        sys = "Answer in Thai where appropriate. Never include explanations outside JSON."
        key = _sha1("LAW|" + snippet[:1200])
        diag = {"route": [], "cached": False}

        # try cache (across providers/models, we key by provider+model+task)
        # We'll attempt providers in order; first hit returns
        for provider, model in [("openai", self.primary_model),
                                ("google", self.fallback1_model),
                                ("openai", self.fallback2_model)]:
            cached = _cache_get(provider, model, "law", key)
            if cached and _validate_json(cached, LAW_SCHEMA):
                diag["route"].append({"provider": provider, "model": model, "used": "cache"})
                diag["cached"] = True
                return cached, diag

        # live calls
        # 1) OpenAI primary
        obj = self._call_openai_json(model=self.primary_model, system=sys, prompt=prompt, schema=LAW_SCHEMA, text=snippet, temperature=0)
        if obj and _validate_json(obj, LAW_SCHEMA):
            _cache_set("openai", self.primary_model, "law", key, obj)
            diag["route"].append({"provider":"openai","model":self.primary_model,"used":"live"})
            return obj, diag

        # 2) Google Gemini fallback
        obj = self._call_gemini_json(model=self.fallback1_model, system=sys, prompt=prompt, schema=LAW_SCHEMA, text=snippet, temperature=0.0)
        if obj and _validate_json(obj, LAW_SCHEMA):
            _cache_set("google", self.fallback1_model, "law", key, obj)
            diag["route"].append({"provider":"google","model":self.fallback1_model,"used":"live"})
            return obj, diag

        # 3) OpenAI GPT-5 final fallback
        obj = self._call_openai_json(model=self.fallback2_model, system=sys, prompt=prompt, schema=LAW_SCHEMA, text=snippet, temperature=0)
        if obj and _validate_json(obj, LAW_SCHEMA):
            _cache_set("openai", self.fallback2_model, "law", key, obj)
            diag["route"].append({"provider":"openai","model":self.fallback2_model,"used":"live"})
            return obj, diag

        diag["route"].append({"provider":"all","model":"failed","used":"none"})
        return None, diag

    # ------------- (B) per-chunk descriptors ------------- #

    def describe_chunk(self, chunk_text: str, section_label: str, breadcrumbs: List[str]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate descriptor JSON for a single chunk (use core_span text).
        """
        context = (section_label or "") + (" — " + " / ".join(breadcrumbs) if breadcrumbs else "")
        prompt = (
            "คุณเป็นผู้ช่วยด้านกฎหมาย ช่วยสรุปย่อเจตนารมณ์ของข้อความกฎหมาย (มาตรา/ข้อ) ด้านล่าง"
            " โดยให้ออกมาเป็น JSON ตามสคีมาที่กำหนดเท่านั้น:\n"
            "- brief: ไม่เกิน 180 ตัวอักษร, ห้ามใส่เลขมาตรา, ให้เน้นใจความ/เงื่อนไข/ข้อยกเว้น\n"
            "- topics: 3–6 คีย์เวิร์ดสำคัญ (ภาษาไทย/คำเฉพาะ)\n"
            "- negations: ถ้ามีข้อความ 'ไม่/ห้าม/เว้นแต่' ให้สรุปสั้น ๆ 0–4 รายการ\n"
            "ห้ามคาดเดาเกินจากเนื้อหา ห้ามอ้างแหล่งอื่น ตอบเป็น JSON ล้วน ๆ เท่านั้น"
        )
        sys = "Return JSON only. Do not include article numbers in brief."
        # limit the text length upstream; still safe to slice here
        snippet = chunk_text[:1200]

        key = _sha1("DESC|" + context + "|" + snippet)
        diag = {"route": [], "cached": False}

        for provider, model in [("openai", self.primary_model),
                                ("google", self.fallback1_model),
                                ("openai", self.fallback2_model)]:
            cached = _cache_get(provider, model, "desc", key)
            if cached and _validate_json(cached, DESC_SCHEMA):
                diag["route"].append({"provider": provider, "model": model, "used": "cache"})
                diag["cached"] = True
                return cached, diag

        # OpenAI primary
        obj = self._call_openai_json(model=self.primary_model, system=sys, prompt=prompt, schema=DESC_SCHEMA, text=snippet, temperature=0.3)
        if obj and _validate_json(obj, DESC_SCHEMA):
            _cache_set("openai", self.primary_model, "desc", key, obj)
            diag["route"].append({"provider":"openai","model":self.primary_model,"used":"live"})
            return obj, diag

        # Gemini fallback
        obj = self._call_gemini_json(model=self.fallback1_model, system=sys, prompt=prompt, schema=DESC_SCHEMA, text=snippet, temperature=0.3)
        if obj and _validate_json(obj, DESC_SCHEMA):
            _cache_set("google", self.fallback1_model, "desc", key, obj)
            diag["route"].append({"provider":"google","model":self.fallback1_model,"used":"live"})
            return obj, diag

        # OpenAI GPT-5
        obj = self._call_openai_json(model=self.fallback2_model, system=sys, prompt=prompt, schema=DESC_SCHEMA, text=snippet, temperature=0.3)
        if obj and _validate_json(obj, DESC_SCHEMA):
            _cache_set("openai", self.fallback2_model, "desc", key, obj)
            diag["route"].append({"provider":"openai","model":self.fallback2_model,"used":"live"})
            return obj, diag

        diag["route"].append({"provider":"all","model":"failed","used":"none"})
        return None, diag

    # batch helper
    def describe_chunks_batch(self, items: List[Tuple[str, str, List[str]]]) -> Tuple[List[Optional[Dict[str,Any]]], Dict[str, Any]]:
        results: List[Optional[Dict[str,Any]]] = []
        log = {"calls": 0, "ok": 0, "fail": 0, "routes": []}
        for (chunk_text, section_label, breadcrumbs) in items:
            t0 = time.time()
            obj, diag = self.describe_chunk(chunk_text, section_label, breadcrumbs)
            dt = round(time.time() - t0, 3)
            log["calls"] += 1
            log["routes"].append({"diag": diag, "latency_s": dt})
            if obj:
                results.append(obj)
                log["ok"] += 1
            else:
                results.append(None)
                log["fail"] += 1
        return results, log

    # ────────── provider calls ────────── #

    def _call_openai_json(self, model: str, system: str, prompt: str, schema: Dict[str, Any], text: str, temperature: float) -> Optional[Dict[str, Any]]:
        try:
            client = _ensure_openai()
            rf = {
                "type": "json_schema",
                "json_schema": {
                    "name": "StrictSchema",
                    "schema": schema,
                    "strict": True,
                },
            }
            msg = f"{prompt}\n\n<document>\n{text}\n</document>"
            resp = client.responses.create(
                model=model,
                input=[{"role": "system", "content": system},
                       {"role": "user", "content": msg}],
                temperature=temperature,
                response_format=rf,
                max_output_tokens=512,
            )
            content = resp.output[0].content[0].text if hasattr(resp, "output") else resp.choices[0].message["content"]  # fallback
            # Some SDK versions return already-parsed JSON in "response"
            try:
                return json.loads(content)
            except Exception:
                # Try tool-style output
                if hasattr(resp, "output") and resp.output and getattr(resp.output[0], "content", None):
                    txt = resp.output[0].content[0].text
                    return json.loads(txt)
            return None
        except Exception:
            return None

    def _call_gemini_json(self, model: str, system: str, prompt: str, schema: Dict[str, Any], text: str, temperature: float) -> Optional[Dict[str, Any]]:
        try:
            genai = _ensure_gemini()
            generation_config = {
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "max_output_tokens": 512,
            }
            m = genai.GenerativeModel(model_name=model, system_instruction=system, generation_config=generation_config)
            msg = f"{prompt}\n\n<document>\n{text}\n</document>"
            resp = m.generate_content([msg])
            if not resp or not getattr(resp, "text", None):
                return None
            return json.loads(resp.text)
        except Exception:
            return None
