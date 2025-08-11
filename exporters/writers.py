# -*- coding: utf-8 -*-
"""
Export helpers
- JSONL (Compat Flat / Rich Meta)
- REPORT.json builder (기존 유지)
"""
from __future__ import annotations
import json
from typing import List, Dict, Any, Optional, Tuple

# ---- 타입 힌트 (런타임 의존 없음) --------------------------------------------
try:
    from parser_core.schema import ParseResult, Chunk  # type: ignore
except Exception:  # pragma: no cover
    ParseResult = Any  # type: ignore
    Chunk = Any        # type: ignore


# ------------------------------------------------------------------------------
# 내부 유틸
# ------------------------------------------------------------------------------

def _meta_common(c: Chunk) -> Dict[str, Any]:
    """청크 메타 공통 필드 정규화"""
    m = dict(c.meta or {})
    return {
        "type": m.get("type") or "article",
        "law_name": m.get("law_name") or "",
        "section_label": m.get("section_label") or "",
        "section_uid": m.get("section_uid") or "",
        "parent_uid": m.get("parent_uid") or "",
        "doc_type": m.get("doc_type") or "",
        "source_file": m.get("source_file") or "",
        # RAG 친화 필드
        "core_span": list(m.get("core_span") or []),
        # UI용/참조
        "breadcrumbs": list(getattr(c, "breadcrumbs", []) or []),
    }


def _record_from_chunk_rich(c: Chunk) -> Dict[str, Any]:
    """Rich Meta용 1 레코드 (감사/디버깅 친화)"""
    meta = _meta_common(c)

    # LLM 사후구조화 결과(있을 때만)
    for k in ("brief", "topics", "negations", "summary_text_raw", "quality"):
        if k in (c.meta or {}):
            meta[k] = c.meta[k]

    # 팩(있을 때만): members 등 메타 보존
    if (c.meta or {}).get("type") == "article_pack":
        for k in ("pack_members", "section_range", "pack_reason", "pack_stats"):
            if k in c.meta:
                meta[k] = c.meta[k]

    return {
        "id": c.meta.get("uid") or c.meta.get("section_uid") or c.meta.get("section_label") or f"chunk_{int(getattr(c,'span_start',0))}_{int(getattr(c,'span_end',0))}",
        "text": c.text,
        "metadata": meta,
    }

# --- Compat 변환 유틸 ----------------------------------------------------------

def _compat_from_chunk(c: Chunk) -> Dict[str, Any]:
    """
    보고서의 compat 예시 스키마에 맞춰 변환:
    {
      "text": "...",
      "span_start": int,
      "span_end": int,
      "meta": {...},
      "breadcrumbs": [...]
    }
    """
    meta = _meta_common(c)

    # compat에서는 breadcrumbs가 top-level에도 있어야 함
    breadcrumbs = meta.pop("breadcrumbs", [])

    # pack 관련 필드는 compat에선 기본적으로 제거(필요시 상위에서 플랫화)
    for k in ("pack_members", "pack_reason", "pack_stats", "section_range"):
        meta.pop(k, None)

    # LLM 생성값이 meta에 있으면 그대로 둔다(brief/topics/negations/quality 등)
    return {
        "text": c.text,
        "span_start": int(getattr(c, "span_start", 0)),
        "span_end": int(getattr(c, "span_end", 0)),
        "meta": meta,
        "breadcrumbs": breadcrumbs,
    }


def _slice_text_from_doc(parse_result: Optional[ParseResult], abs_span: Optional[Tuple[int, int]], pack_text: Optional[str], rel_span: Optional[Tuple[int, int]], pack_core_offset: Optional[int]) -> str:
    """
    멤버 텍스트를 안전하게 잘라서 반환.
    우선순위: 절대 스팬 → (팩 텍스트 + 상대 스팬) → 빈 문자열
    """
    try:
        if parse_result and abs_span and isinstance(abs_span[0], int) and isinstance(abs_span[1], int):
            s, e = abs_span
            if 0 <= s < e <= len(parse_result.full_text):
                return parse_result.full_text[s:e]
    except Exception:
        pass

    try:
        if pack_text is not None and rel_span and isinstance(rel_span[0], int) and isinstance(rel_span[1], int):
            a, b = rel_span
            if isinstance(pack_core_offset, int) and pack_core_offset > 0:
                a += pack_core_offset
                b += pack_core_offset
            if 0 <= a < b <= len(pack_text):
                return pack_text[a:b]
    except Exception:
        pass

    return ""


def _explode_pack_to_flat_compat(
    c: Chunk,
    parse_result: Optional[ParseResult]
) -> List[Dict[str, Any]]:
    """
    article_pack → 멤버 단위 '일반 청크(compat)' 레코드로 분해
    """
    items: List[Dict[str, Any]] = []
    pack_meta = c.meta or {}
    members = pack_meta.get("pack_members") or pack_meta.get("members") or []
    if not members:
        # 멤버 정보가 없으면 팩 자체를 일반 청크처럼 내보낸다(최소 호환).
        items.append(_compat_from_chunk(c))
        return items

    # 팩 공통 정보
    base_meta = _meta_common(c)
    breadcrumbs = base_meta.get("breadcrumbs", [])
    base_meta.pop("breadcrumbs", None)

    pack_core_span = (pack_meta.get("core_span") or base_meta.get("core_span") or [0, 0])
    pack_core_offset = 0
    if isinstance(pack_core_span, (list, tuple)) and len(pack_core_span) == 2:
        try:
            pack_core_offset = int(pack_core_span[0]) - int(getattr(c, "span_start", 0))
        except Exception:
            pack_core_offset = 0

    for i, m in enumerate(members, start=1):
        label = m.get("label") or m.get("section_label") or f"member_{i}"
        abs_span = None
        rel_span = None
        if isinstance(m.get("span"), (list, tuple)) and len(m["span"]) == 2:
            abs_span = (int(m["span"][0]), int(m["span"][1]))
        if isinstance(m.get("offset_in_pack"), (list, tuple)) and len(m["offset_in_pack"]) == 2:
            rel_span = (int(m["offset_in_pack"][0]), int(m["offset_in_pack"][1]))

        member_text = _slice_text_from_doc(parse_result, abs_span, c.text, rel_span, pack_core_offset) or c.text

        meta = dict(base_meta)
        meta.update({
            "type": "article",
            "section_label": label,
            "section_uid": m.get("uid") or m.get("section_uid") or "",
            "pack_origin": (pack_meta.get("uid") or pack_meta.get("section_uid") or ""),
            "pack_section_range": pack_meta.get("section_range") or "",
        })
        # 팩 레벨 LLM 메타가 있으면 복사(선택)
        for k in ("brief", "topics", "negations", "quality"):
            if k in pack_meta:
                meta[k] = pack_meta[k]

        items.append({
            "text": member_text,
            "span_start": int(m.get("span", [getattr(c,"span_start",0)])[0]) if isinstance(m.get("span"), list) else int(getattr(c,"span_start",0)),
            "span_end": int(m.get("span", [0, getattr(c,"span_end",0)])[1]) if isinstance(m.get("span"), list) else int(getattr(c,"span_end",0)),
            "meta": meta,
            "breadcrumbs": breadcrumbs,
        })

    return items


# ------------------------------------------------------------------------------
# 공개 API: JSONL
# ------------------------------------------------------------------------------

def to_jsonl_rich_meta(chunks: List[Chunk]) -> str:
    """
    팩/LLM 메타를 그대로 보존한 JSONL (RAG + 감사/디버깅에 유리)
    """
    out = []
    for c in chunks:
        rec = _record_from_chunk_rich(c)
        out.append(json.dumps(rec, ensure_ascii=False))
    return "\n".join(out)


def to_jsonl_compat_flat(chunks: List[Chunk], parse_result: Optional[ParseResult] = None) -> str:
    """
    표준 RAG 파이프라인 친화(JSON만 보고도 동작).
    - article_pack을 멤버별 '일반 청크'로 분해
    - 일반 청크는 compat 스키마로 직렬화
    """
    out: List[str] = []
    for c in chunks:
        ctype = (c.meta or {}).get("type") or "article"
        if ctype == "article_pack":
            for rec in _explode_pack_to_flat_compat(c, parse_result):
                out.append(json.dumps(rec, ensure_ascii=False))
        else:
            rec = _compat_from_chunk(c)
            out.append(json.dumps(rec, ensure_ascii=False))
    return "\n".join(out)


# ------------------------------------------------------------------------------
# REPORT.json (기존 함수 유지)  -----------------------------------------------
# ------------------------------------------------------------------------------

def make_debug_report(
    *,
    parse_result: ParseResult,
    chunks: List[Chunk],
    source_file: str,
    law_name: str,
    run_config: Dict[str, Any],
    debug: Dict[str, Any]
) -> str:
    """
    디버그/품질 진단용 REPORT.json 문자열
    """
    import statistics as stats

    # 길이 통계
    lens = [len(c.text) for c in chunks]
    lens_sorted = sorted(lens)
    p50 = lens_sorted[len(lens_sorted)//2] if lens_sorted else 0
    p75 = lens_sorted[int(len(lens_sorted)*0.75)] if lens_sorted else 0
    top_long = sorted(
        [{"section": (c.meta or {}).get("section_label",""), "len": len(c.text)} for c in chunks],
        key=lambda x: -x["len"]
    )[:5]
    top_short = sorted(
        [{"section": (c.meta or {}).get("section_label",""), "len": len(c.text)} for c in chunks],
        key=lambda x: x["len"]
    )[:5]

    obj = {
        "file": source_file,
        "doc_type": (parse_result.doc_type or "unknown"),
        "law_name": law_name,
        "chunks": len(chunks),
        "coverage": debug.get("coverage_calc",{}).get("coverage", 0.0),
        "length_stats": {
            "p50": p50,
            "p75": p75,
            "max": max(lens) if lens else 0,
            "min": min(lens) if lens else 0,
            "top_longest": top_long,
            "top_shortest": top_short
        },
        "run_config": run_config,
        "debug": debug,
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)
