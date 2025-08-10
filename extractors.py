# -*- coding: utf-8 -*-
"""
LLM 프리포맷 요약 → 규칙 기반 사후 구조화
- normalize_text, split_bullets
- extract_brief / extract_topics / extract_negations
- build_summary_record: app에서 쓰는 최종 패키저
태국어 특화 간단 규칙(외부 형태소 분석기 없이 동작).
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple

THAI_SPACE = r"\u0E00-\u0E7F"  # basic thai block
RE_MULTI_NL = re.compile(r"\n{2,}")
RE_SPACES = re.compile(r"[ \t]{2,}")
RE_BRACKETS_NOTE = re.compile(r"\[[^\]]{1,40}\]")        # [1], [อ้างอิง]
RE_FOOTNOTE = re.compile(r"\(\s*footnote.*?\)", re.I)
RE_DUP_PUNCT = re.compile(r"([,;:])\1+")
RE_BULLET_PREFIX = re.compile(r"^\s*(?:[-•–—]|\(\d+\)|\d+[\)\.]|[๑-๙]+[\)\.]|●|▪︎|▶︎)\s*")
RE_ENUM_INLINE = re.compile(r"\s+[;•]\s+")
RE_THAI_DIGIT = str.maketrans("๑๒๓๔๕๖๗๘๙", "123456789")
RE_NUM_BRACKET = re.compile(r"\(\d+\)")

TH_STOP = set(["การ","ผู้","ความ","ของ","และ","หรือ","ที่","เพื่อ","ให้","ตาม","แก่","ซึ่ง","ใน","เป็น","ได้","ต้อง","ห้าม","เว้นแต่","รวมทั้ง","กับ","โดย","ต่อ","แก่","จาก","จนถึง","ถึง","ดังกล่าว"])
NEG_KEYS = ["ไม่","มิ","ห้าม","เว้นแต่","ยกเว้น","ต้องไม่","ห้ามมิให้"]

def normalize_text(s: str) -> str:
    s = s.replace("\r", "")
    s = RE_FOOTNOTE.sub("", s)
    s = RE_BRACKETS_NOTE.sub("", s)
    s = RE_DUP_PUNCT.sub(r"\1", s)
    s = RE_MULTI_NL.sub("\n", s)
    s = RE_SPACES.sub(" ", s)
    s = s.translate(RE_THAI_DIGIT)
    return s.strip()

def split_bullets(s: str) -> List[str]:
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    items: List[str] = []
    for ln in lines:
        ln = RE_NUM_BRACKET.sub("", ln)
        if RE_BULLET_PREFIX.match(ln):
            ln = RE_BULLET_PREFIX.sub("", ln)
            items.append(ln.strip())
        else:
            # inline bullets e.g., "…; … ; …" or "… • …"
            sub = [x.strip() for x in RE_ENUM_INLINE.split(ln) if x.strip()]
            if len(sub) > 1:
                items.extend(sub)
            else:
                items.append(ln)
    # 중복 제거 + 길이 필터
    seen = set()
    uniq = []
    for it in items:
        if len(it) < 3: 
            continue
        key = it
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq

def _cut_smart(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    # 문장 경계 기준 우선
    cut = s[:max_len]
    p = max(cut.rfind(" "), cut.rfind("।"), cut.rfind("."), cut.rfind(")"), cut.rfind("]"))
    if p > max_len * 0.6:
        return cut[:p].rstrip(" ,;:") + "…"
    return cut.rstrip(" ,;:") + "…"

def extract_brief(bullets: List[str], max_len: int = 180) -> str:
    if not bullets:
        return ""
    cand = bullets[0]
    cand = re.sub(r"^[\"'“”‘’\(\[]+|[\"'“”‘’\)\]]+$", "", cand).strip()
    # 군더더기 접미사 제거
    cand = re.sub(r"(?:ตาม.*$|ดังกล่าว.*$)", "", cand).strip() or bullets[0]
    return _cut_smart(cand, max_len)

def _ngrams(words: List[str], n: int) -> List[str]:
    return [" ".join(words[i:i+n]) for i in range(0, len(words)-n+1)]

def _tokenize_thai_words(s: str) -> List[str]:
    # 아주 단순 토크나이저: 공백/구두점 기준
    s = re.sub(r"[,\.;:()\[\]{}<>/|«»“”‘’\"']", " ", s)
    ws = [w for w in s.split() if w]
    return ws

def extract_topics(bullets: List[str], k_min: int = 3, k_max: int = 6) -> List[str]:
    if not bullets:
        return []
    text = " ; ".join(bullets[:6])
    # 쉼표/접속사 분할 우선
    parts = re.split(r"\s*(?:,|;|และ|หรือ|รวมทั้ง|กับ)\s*", text)
    cand: List[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 4:
            continue
        ws = [w for w in _tokenize_thai_words(p) if w not in TH_STOP]
        if not ws:
            continue
        # n-gram 1~3
        grams = ws[:]
        for n in (2,3):
            grams += _ngrams(ws, n)
        grams = [g for g in grams if 4 <= len(g) <= 24]
        # 가장 길고 정보량 있어 보이는 후보 몇 개만
        grams.sort(key=lambda x: (-len(x), x))
        for g in grams[:2]:
            cand.append(g)

    # 정제: 중복/부분문구 제거
    uniq: List[str] = []
    for x in cand:
        if any(x != y and x in y for y in cand):
            continue
        if x not in uniq:
            uniq.append(x)
    uniq = [x for x in uniq if not any(x != y and x in y for y in uniq)]
    return uniq[:max(k_min, min(k_max, len(uniq)))]

def extract_negations(s: str, max_items: int = 4, window: int = 56) -> List[str]:
    if not s:
        return []
    hits: List[str] = []
    for key in NEG_KEYS:
        for m in re.finditer(re.escape(key), s):
            a = max(0, m.start() - window)
            b = min(len(s), m.end() + window)
            frag = s[a:b].strip()
            hits.append(frag)
    # 중복/유사 병합
    uniq = []
    for h in hits:
        if any(h in u or u in h for u in uniq):
            continue
        uniq.append(h)
    return uniq[:max_items]

def build_summary_record(
    law_name: str,
    section_label: str,
    breadcrumbs: List[str],
    span: Tuple[int,int],
    llm_text: str,
    *,
    brief_max_len: int = 180
) -> Dict[str, Any]:
    norm = normalize_text(llm_text)
    bullets = split_bullets(norm)
    brief = extract_brief(bullets, max_len=brief_max_len)
    topics = extract_topics(bullets)
    negs = extract_negations(norm)

    flags: List[str] = []
    if len(brief) < 20: flags.append("brief_too_short")
    if len(brief) > brief_max_len: flags.append("brief_truncated")
    if len(topics) < 2: flags.append("topics_low")
    # 태국어 비율 간단 체크
    thai_chars = sum(1 for ch in norm if "\u0E00" <= ch <= "\u0E7F")
    if thai_chars < max(30, int(len(norm)*0.5)):
        flags.append("thai_ratio_low")

    return {
        "law_name": law_name,
        "section_label": section_label,
        "breadcrumbs": breadcrumbs,
        "span": [int(span[0]), int(span[1])],
        "brief": brief,
        "topics": topics,
        "negations": negs,
        "summary_text_raw": norm,
        "quality": {
            "brief_len": len(brief),
            "topics_n": len(topics),
            "has_negation": bool(negs),
            "flags": flags
        }
    }
