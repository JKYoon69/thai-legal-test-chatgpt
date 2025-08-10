# -*- coding: utf-8 -*-
"""
postprocess.py  (self-contained, app.py와 호환되는 경량 청크 반환)

- 텍스트만으로 조문 단위 청크 생성/보정
- 옵션:
  * strict_lossless: 커버리지 1.0 목표 gap-sweeper
  * split_long_articles: 롱 조문 분할(문장 경계 우선, 실패 시 soft cut)
  * tail_merge_min_chars: 분할 마지막 자투리(≤임계) 앞 파트로 흡수
  * overlap_chars: 분할 시 오버랩(표시 범위 span은 포함, core_span은 제외)
  * skip_short_chars: 너무 짧은 조문(기본 180자) micro-join으로 앞 기사에 흡수
  * include_front_matter / include_headnotes / include_gap_fallback
- 공개 함수:
  validate_tree, repair_tree, guess_law_name, make_chunks, merge_small_trailing_parts
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# -----------------------------------------------------------------------------
# 경량 Chunk (app.py / exporters가 기대하는 속성 이름과 동일)
# -----------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    span_start: int
    span_end: int
    meta: Dict[str, Any]
    breadcrumbs: List[str]


# -----------------------------------------------------------------------------
# 내부 유틸
# -----------------------------------------------------------------------------
def _get_full_text(result) -> str:
    try:
        if isinstance(result.full_text, str):
            return result.full_text
    except Exception:
        pass
    return getattr(result, "text", "") or ""


def _safe_slice(s: str, a: int, b: int) -> str:
    a = max(0, min(len(s), int(a)))
    b = max(0, min(len(s), int(b)))
    if b <= a:
        return ""
    return s[a:b]


def _mk_uid(label: str, start: int) -> str:
    return f"{label}|{start}"


# -----------------------------------------------------------------------------
# 라벨/패턴
# -----------------------------------------------------------------------------
THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"

ARTICLE_PAT = re.compile(r"(มาตรา)\s*([0-9" + THAI_DIGITS + r"]+)")
HEAD_PAT = re.compile(
    r"(?m)^(?P<label>\s*(ภาค|ลักษณะ|หมวด|ส่วน|บท(?:บัญญัติทั่วไป)?)[^\n]{0,80})\s*$"
)

CANDIDATE_BREAK = re.compile(r"[.!?;:]\s|\n{1,3}|[”\"'’)]\s|ฯ")
PART_RE = re.compile(r"^(?P<base>.+?)\s*\(part\s*(?P<idx>\d+)\)\s*$", re.IGNORECASE)


def _part_info(label: str) -> Tuple[str, Optional[int]]:
    if not isinstance(label, str):
        return ("", None)
    m = PART_RE.match(label.strip())
    if not m:
        return (label.strip(), None)
    try:
        return (m.group("base").strip(), int(m.group("idx")))
    except Exception:
        return (label.strip(), None)


# -----------------------------------------------------------------------------
# validate / repair / guess
# -----------------------------------------------------------------------------
def validate_tree(result) -> List[str]:
    issues: List[str] = []
    text = _get_full_text(result)
    if not text:
        issues.append("empty_full_text")
        return issues
    if not ARTICLE_PAT.search(text):
        issues.append("no_article_pattern_found")
    return issues


def repair_tree(result) -> Dict[str, Any]:
    return {"repaired": False, "note": "text-only light repair; no structural changes"}


def guess_law_name(text: str) -> str:
    if not text:
        return ""
    first_block = "\n".join(text.splitlines()[:12])
    for line in first_block.splitlines():
        t = line.strip()
        if not t:
            continue
        if any(k in t for k in ("พระราชบัญญัติ", "ประมวลกฎหมาย", "รัฐธรรมนูญ", "พระราชกำหนด", "ประกาศ")):
            if len(t) < 12:
                continue
            return t
    for line in first_block.splitlines():
        if line.strip():
            return line.strip()
    return ""


# -----------------------------------------------------------------------------
# 스캔/세그먼트
# -----------------------------------------------------------------------------
def _scan_markers(text: str) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    heads: List[Tuple[int, int, str]] = []
    arts: List[Tuple[int, int, str]] = []

    for m in HEAD_PAT.finditer(text):
        s, e = m.span()
        label = m.group("label").strip()
        if 1 <= len(label) <= 120:
            heads.append((s, e, label))

    for m in ARTICLE_PAT.finditer(text):
        s, e = m.span()
        label = (m.group(1) + " " + m.group(2)).strip()
        arts.append((s, e, label))

    heads.sort(key=lambda x: x[0])
    arts.sort(key=lambda x: x[0])
    return heads, arts


def _collect_segments(text: str, heads, arts) -> List[Dict[str, Any]]:
    points = []
    for s, e, lab in heads:
        points.append((s, "headnote", (s, e, lab)))
    for s, e, lab in arts:
        points.append((s, "article", (s, e, lab)))
    points.sort(key=lambda x: x[0])

    segs: List[Dict[str, Any]] = []
    breadcrumbs: List[str] = []

    for i, (_, typ, payload) in enumerate(points):
        start_of_this = payload[0]
        label = payload[2]
        end_of_this = points[i + 1][0] if i + 1 < len(points) else len(text)

        if typ == "headnote":
            breadcrumbs.append(label)
            segs.append({"type": "headnote", "label": label, "span": (start_of_this, end_of_this), "head_ctx": list(breadcrumbs)})
        else:
            segs.append({"type": "article", "label": label, "span": (start_of_this, end_of_this), "head_ctx": list(breadcrumbs)})

    return segs


# -----------------------------------------------------------------------------
# 롱 조문 분할
# -----------------------------------------------------------------------------
def _best_cut(text: str, near: int, left: int, right: int) -> int:
    if left >= right:
        return near
    window = text[left:right]
    best = None
    best_dist = 10**9
    for m in CANDIDATE_BREAK.finditer(window):
        cut = left + m.end()
        dist = abs(cut - near)
        if dist < best_dist:
            best = cut
            best_dist = dist
    return best if best is not None else near


def _split_article_to_cores(
    text: str, s: int, e: int, limit: int, soft_cut: bool
) -> List[Tuple[int, int]]:
    """
    코어(core) 단위로 먼저 자른다. (오버랩은 나중에 span에서만 적용)
    return: [(core_s, core_e), ...]
    """
    if e - s <= limit:
        return [(s, e)]

    cores: List[Tuple[int, int]] = []
    cursor = s
    while cursor < e:
        target = min(cursor + limit, e)
        cut = _best_cut(text, target, cursor + int(limit * 0.6), min(e, cursor + int(limit * 1.4))) if soft_cut else target
        cut = max(cursor + 1, min(cut, e))
        cores.append((cursor, cut))
        cursor = cut

    return cores


def _cores_to_parts_with_overlap(
    cores: List[Tuple[int, int]],
    article_bounds: Tuple[int, int],
    label: str,
    overlap: int
) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    """
    core 목록을 받아 오버랩을 포함한 표시 범위(span)와 core_span을 만들어낸다.
    - 첫 파트는 좌측 오버랩 0
    - 마지막 파트는 우측 오버랩 0
    """
    parts: List[Tuple[str, Tuple[int, int], Tuple[int, int]]] = []
    art_s, art_e = article_bounds

    for idx, (cs, ce) in enumerate(cores, start=1):
        left_overlap = overlap if idx > 1 else 0
        right_overlap = overlap if idx < len(cores) else 0

        span_s = max(art_s, cs - left_overlap)
        span_e = min(art_e, ce + right_overlap)

        parts.append((f"{label} (part {idx})", (span_s, span_e), (cs, ce)))

    return parts


def _apply_tail_merge_to_parts(
    parts: List[Tuple[str, Tuple[int, int], Tuple[int, int]]],
    tail_merge_min_chars: int
) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    """
    마지막 파트(span/core 모두)를 앞 파트로 흡수(조건: 마지막 core 길이 <= 임계)
    """
    if len(parts) >= 2:
        last_l, (ls, le), (lcs, lce) = parts[-1]
        prev_l, (ps, pe), (pcs, pce) = parts[-2]
        if (lce - lcs) <= tail_merge_min_chars:
            # 앞 파트의 표시/코어 끝을 확장
            parts[-2] = (prev_l, (ps, max(pe, le)), (pcs, lce))
            parts.pop(-1)
    return parts


# -----------------------------------------------------------------------------
# 청크 생성
# -----------------------------------------------------------------------------
def make_chunks(
    *,
    result,
    mode: str = "article_only",
    source_file: str = "",
    law_name: str = "",
    include_front_matter: bool = True,
    include_headnotes: bool = True,
    include_gap_fallback: bool = True,
    allowed_headnote_levels: Optional[List[str]] = None,
    min_headnote_len: int = 24,
    min_gap_len: int = 24,
    strict_lossless: bool = True,
    split_long_articles: bool = True,
    split_threshold_chars: int = 1500,
    tail_merge_min_chars: int = 200,
    overlap_chars: int = 200,
    skip_short_chars: int = 180,
    soft_cut: bool = True,
) -> Tuple[List[Chunk], Dict[str, Any]]:
    """
    텍스트 기반 조문 청킹(라이트). full_text만으로 동작.
    반환: (chunks, diagnostics)
    """
    text = _get_full_text(result)
    N = len(text)
    chunks: List[Chunk] = []
    diag: Dict[str, Any] = {
        "heads": 0,
        "articles": 0,
        "split_parts": 0,
        "gap_fallback": 0,
        "removed_short": 0,
        "micro_joined": 0,
    }

    if N == 0:
        return ([], {"error": "empty_text"})

    heads, arts = _scan_markers(text)
    segs = _collect_segments(text, heads, arts)

    # 1) front_matter
    first_pos = min([s for s, _, _ in heads + arts], default=None)
    if include_front_matter and first_pos and first_pos > 0:
        fm = Chunk(
            text=_safe_slice(text, 0, first_pos),
            span_start=0, span_end=first_pos,
            meta={
                "type": "front_matter",
                "law_name": law_name,
                "source_file": source_file,
                "section_label": "front_matter",
                "section_uid": _mk_uid("front_matter", 0),
                "core_span": [0, first_pos],
                "doc_type": getattr(result, "doc_type", "") or "",
            },
            breadcrumbs=[]
        )
        chunks.append(fm)

    # 2) segments → chunks
    for seg in segs:
        typ = seg["type"]
        s, e = seg["span"]
        label = seg["label"].strip()
        bcs = [b.strip() for b in seg.get("head_ctx") or []]

        if typ == "headnote":
            if not include_headnotes:
                continue
            if len(_safe_slice(text, s, e).strip()) < min_headnote_len:
                continue
            ch = Chunk(
                text=_safe_slice(text, s, e),
                span_start=s, span_end=e,
                meta={
                    "type": "headnote",
                    "law_name": law_name,
                    "source_file": source_file,
                    "section_label": label,
                    "section_uid": _mk_uid(label, s),
                    "core_span": [s, e],
                    "doc_type": getattr(result, "doc_type", "") or "",
                },
                breadcrumbs=bcs
            )
            chunks.append(ch)
            diag["heads"] += 1
        else:
            # article
            if not (0 <= s < e <= N):
                continue

            if split_long_articles and (e - s) > split_threshold_chars:
                cores = _split_article_to_cores(text, s, e, split_threshold_chars, soft_cut)
                parts = _cores_to_parts_with_overlap(cores, (s, e), label, overlap_chars)
                parts = _apply_tail_merge_to_parts(parts, tail_merge_min_chars)
                diag["split_parts"] += len(parts)

                for lab, (ps, pe), (cs, ce) in parts:
                    ch = Chunk(
                        text=_safe_slice(text, ps, pe),
                        span_start=ps, span_end=pe,
                        meta={
                            "type": "article",
                            "law_name": law_name,
                            "source_file": source_file,
                            "section_label": lab,
                            "section_uid": _mk_uid(lab, ps),
                            "core_span": [cs, ce],
                            "doc_type": getattr(result, "doc_type", "") or "",
                        },
                        breadcrumbs=bcs
                    )
                    chunks.append(ch)
            else:
                ch = Chunk(
                    text=_safe_slice(text, s, e),
                    span_start=s, span_end=e,
                    meta={
                        "type": "article",
                        "law_name": law_name,
                        "source_file": source_file,
                        "section_label": label,
                        "section_uid": _mk_uid(label, s),
                        "core_span": [s, e],
                        "doc_type": getattr(result, "doc_type", "") or "",
                    },
                    breadcrumbs=bcs
                )
                chunks.append(ch)
            diag["articles"] += 1

    # 3) 정렬 보장
    chunks.sort(key=lambda c: (int(c.span_start), int(c.span_end)))

    # 4) micro-join: 너무 짧은 article은 앞 article로 흡수
    def _is_article(ch: Chunk) -> bool:
        return ch.meta.get("type") == "article"

    i = 1
    while i < len(chunks):
        curr = chunks[i]
        if _is_article(curr) and len(curr.text.strip()) < int(skip_short_chars):
            # 앞 기사와 동일 level(crumbs)일 때만 흡수
            prev = chunks[i - 1]
            if _is_article(prev) and prev.breadcrumbs == curr.breadcrumbs:
                # prev 확장 (표시/코어 모두)
                prev_core = prev.meta.get("core_span", [prev.span_start, prev.span_end])
                curr_core = curr.meta.get("core_span", [curr.span_start, curr.span_end])
                prev.meta["core_span"] = [int(prev_core[0]), int(curr_core[1])]
                prev.span_end = int(curr.span_end)
                prev.text = _safe_slice(text, int(prev.span_start), int(prev.span_end))
                # 현재 제거
                del chunks[i]
                diag["micro_joined"] += 1
                continue
        i += 1

    # 5) gap sweeper (무손실 옵션)
    if strict_lossless and include_gap_fallback:
        filled: List[Chunk] = []
        cur = 0
        for ch in chunks:
            s, e = int(ch.span_start), int(ch.span_end)
            if s > cur and (s - cur) >= min_gap_len:
                gf = Chunk(
                    text=_safe_slice(text, cur, s),
                    span_start=cur, span_end=s,
                    meta={
                        "type": "gap_fallback",
                        "law_name": law_name,
                        "source_file": source_file,
                        "section_label": "gap",
                        "section_uid": _mk_uid("gap", cur),
                        "core_span": [cur, s],
                        "doc_type": getattr(result, "doc_type", "") or "",
                    },
                    breadcrumbs=[]
                )
                filled.append(gf)
                diag["gap_fallback"] += 1
            filled.append(ch)
            cur = max(cur, e)
        if cur < N and (N - cur) >= min_gap_len:
            gf = Chunk(
                text=_safe_slice(text, cur, N),
                span_start=cur, span_end=N,
                meta={
                    "type": "gap_fallback",
                    "law_name": law_name,
                    "source_file": source_file,
                    "section_label": "gap",
                    "section_uid": _mk_uid("gap", cur),
                    "core_span": [cur, N],
                    "doc_type": getattr(result, "doc_type", "") or "",
                },
                breadcrumbs=[]
            )
            filled.append(gf)
            diag["gap_fallback"] += 1
        chunks = filled

    return (chunks, diag)


# -----------------------------------------------------------------------------
# tail-sweep: 마지막 짧은 파트는 앞 파트로 병합 (보수적)
# -----------------------------------------------------------------------------
def merge_small_trailing_parts(
    chunks: List[Chunk],
    *,
    full_text: str,
    max_tail_chars: int = 200
) -> Dict[str, Any]:
    if not chunks:
        return {"merged_count": 0, "affected_sections": []}

    chunks.sort(key=lambda c: (int(c.span_start), int(c.span_end)))

    merged = 0
    affected: List[str] = []
    i = 1
    while i < len(chunks):
        prev = chunks[i - 1]
        curr = chunks[i]

        if (prev.meta.get("type") == "article" and curr.meta.get("type") == "article"):
            prev_label = prev.meta.get("section_label", "")
            curr_label = curr.meta.get("section_label", "")
            prev_base, prev_idx = _part_info(prev_label)
            curr_base, curr_idx = _part_info(curr_label)

            is_series = (
                prev_base and curr_base and prev_base == curr_base and
                (prev_idx is not None) and (curr_idx is not None) and (curr_idx == prev_idx + 1)
            )
            # "마지막 파트"에 대한 보수적 병합: 뒤 파트 길이가 매우 짧을 때만
            curr_core = curr.meta.get("core_span", [curr.span_start, curr.span_end])
            curr_len = int(curr_core[1]) - int(curr_core[0])

            if is_series and curr_len <= int(max_tail_chars):
                prev_core = prev.meta.get("core_span", [prev.span_start, prev.span_end])
                prev.meta["core_span"] = [int(prev_core[0]), int(curr_core[1])]
                prev.span_end = int(curr.span_end)
                prev.text = _safe_slice(full_text, int(prev.span_start), int(prev.span_end))

                del chunks[i]
                merged += 1
                affected.append(prev_base)
                continue
        i += 1

    return {"merged_count": merged, "affected_sections": affected}
