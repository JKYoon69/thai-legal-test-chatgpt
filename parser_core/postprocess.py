# -*- coding: utf-8 -*-
"""
postprocess.py  (self-contained, app.py와 호환되는 경량 청크 반환)
- 파서 결과(ParseResult)에서 조문 단위 청크 생성/보정
- 옵션:
  * strict_lossless: 커버리지 1.0 목표 gap-sweeper
  * split_long_articles: 롱 조문 분할(문장 경계 우선, 실패 시 soft cut)
  * tail_merge_min_chars: 분할 마지막 자투리(≤임계) 앞 파트로 흡수
  * overlap_chars: 분할 시 앞/뒤 오버랩(문맥)
  * include_front_matter / include_headnotes / include_gap_fallback
- 공개 함수:
  validate_tree, repair_tree, guess_law_name, make_chunks, merge_small_trailing_parts
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

# -----------------------------------------------------------------------------
# 경량 Chunk (app.py / exporters에서 기대하는 속성들만 제공)
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
    """ParseResult로부터 본문 텍스트를 안전하게 얻는다."""
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
    """'มาตรา 119 (part 2)' -> ('มาตรา 119', 2) / 'มาตรา 119' -> ('มาตรา 119', None)"""
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
    """
    간단 진단:
    - full_text 비어있는지
    - 'มาตรา' 존재하는지
    """
    issues: List[str] = []
    text = _get_full_text(result)
    if not text:
        issues.append("empty_full_text")
        return issues
    if not ARTICLE_PAT.search(text):
        issues.append("no_article_pattern_found")
    return issues


def repair_tree(result) -> Dict[str, Any]:
    """
    구조 수복(라이트). 텍스트 기반으로 보정 포인트만 리턴.
    실제 트리 재배열은 추후 확장 가능.
    """
    return {"repaired": False, "note": "text-only light repair; no structural changes"}


def guess_law_name(text: str) -> str:
    """
    법령명 추정(휴리스틱):
    - 첫 10~12줄에서 'พระราชบัญญัติ', 'ประมวลกฎหมาย', 'รัฐธรรมนูญ' 등 키워드 라인 추출
    """
    if not text:
        return ""
    first_block = "\n".join(text.splitlines()[:12])
    for line in first_block.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        if any(k in line_strip for k in ("พระราชบัญญัติ", "ประมวลกฎหมาย", "รัฐธรรมนูญ", "พระราชกำหนด", "ประกาศ")):
            if len(line_strip) < 12:
                continue
            return line_strip
    for line in first_block.splitlines():
        if line.strip():
            return line.strip()
    return ""


# -----------------------------------------------------------------------------
# 스캔/세그먼트
# -----------------------------------------------------------------------------
def _scan_markers(text: str) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
    """
    텍스트에서 헤드노트/조문 시작 위치를 스캔.
    return: (heads, arts)
      heads = [(start, end, label), ...]
      arts  = [(start, end, label), ...]  # end는 라인 끝(라벨 영역)
    """
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
    """
    head/article 시작점 기준으로 다음 시작점 직전까지를 segment로 본다.
    각 segment: {"type": "headnote"|"article", "label": str, "span": (s,e), "head_ctx": [...breadcrumbs...]}
    """
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
    """[left, right] 내에서 near에 가까운 문장 경계 후보를 찾는다. 실패 시 near 반환."""
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


def _split_article(text: str, s: int, e: int, label: str, overlap: int, limit: int,
                   tail_merge_min_chars: int, soft_cut: bool) -> List[Tuple[str, Tuple[int, int], Tuple[int, int]]]:
    """
    롱 조문 분할.
    return: [(label_or_part, (span_start, span_end), (core_start, core_end)), ...]
    """
    length = e - s
    if length <= limit:
        return [(label, (s, e), (s, e))]

    parts = []
    cursor = s
    idx = 1
    while cursor < e:
        target = min(cursor + limit, e)
        cut = _best_cut(text, target, cursor + int(limit * 0.6), min(e, cursor + int(limit * 1.4))) if soft_cut else target
        cut = max(cursor + 1, min(cut, e))
        span_s = cursor
        span_e = cut

        core_s = span_s
        core_e = span_e
        if overlap > 0:
            if parts:
                core_s = span_s + min(overlap, (span_e - span_s) // 3)
            if span_e < e:
                core_e = span_e - min(overlap, (span_e - span_s) // 3)

        parts.append((f"{label} (part {idx})", (span_s, span_e), (core_s, core_e)))
        cursor = span_e
        idx += 1

    # tail 병합(마지막 조각이 너무 짧으면 앞 파트로 합치기)
    if len(parts) >= 2:
        last_label, (ls, le), (lcs, lce) = parts[-1]
        prev_label, (ps, pe), (pcs, pce) = parts[-2]
        if (le - ls) <= tail_merge_min_chars:
            parts[-2] = (prev_label, (ps, le), (pcs, le))
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
    soft_cut: bool = True,
) -> Tuple[List[Chunk], Dict[str, Any]]:
    """
    텍스트 기반 조문 청킹(라이트). 파서 트리가 없어도 full_text만으로 동작.
    반환: (chunks, diagnostics)
    """
    text = _get_full_text(result)
    N = len(text)
    chunks: List[Chunk] = []
    diag: Dict[str, Any] = {"heads": 0, "articles": 0, "split_parts": 0, "gap_fallback": 0}

    if N == 0:
        return ([], {"error": "empty_text"})

    heads, arts = _scan_markers(text)
    segs = _collect_segments(text, heads, arts)

    # front_matter
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

    # segments → chunks
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
                parts = _split_article(
                    text, s, e, label,
                    overlap=overlap_chars,
                    limit=split_threshold_chars,
                    tail_merge_min_chars=tail_merge_min_chars,
                    soft_cut=soft_cut
                )
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

    # 정렬 보장
    chunks.sort(key=lambda c: (int(c.span_start), int(c.span_end)))

    # gap sweeper (무손실 옵션)
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
# tail-sweep: 마지막 짧은 파트는 앞 파트로 병합
# -----------------------------------------------------------------------------
def merge_small_trailing_parts(
    chunks: List[Chunk],
    *,
    full_text: str,
    max_tail_chars: int = 200
) -> Dict[str, Any]:
    """
    분할 시리즈의 마지막(part N) 조각이 너무 짧으면 앞 조각으로 흡수.
    - 같은 시리즈(base) 연속(part n → n+1)
    - 뒤 파트 길이 ≤ max_tail_chars
    - prev.meta.core_span[1], prev.span_end를 curr 끝까지 확장, curr 제거
    """
    if not chunks:
        return {"merged_count": 0, "affected_sections": []}

    chunks.sort(key=lambda c: (int(c.span_start), int(c.span_end)))

    merged = 0
    affected: List[str] = []
    i = 1
    while i < len(chunks):
        prev = chunks[i - 1]
        curr = chunks[i]

        if (getattr(prev, "meta", {}).get("type") == "article" and
            getattr(curr, "meta", {}).get("type") == "article"):

            prev_label = prev.meta.get("section_label", "")
            curr_label = curr.meta.get("section_label", "")
            prev_base, prev_idx = _part_info(prev_label)
            curr_base, curr_idx = _part_info(curr_label)

            is_series = (
                prev_base and curr_base and prev_base == curr_base and
                (prev_idx is not None) and (curr_idx is not None) and (curr_idx == prev_idx + 1)
            )

            curr_len = int(curr.span_end) - int(curr.span_start)

            if is_series and curr_len <= int(max_tail_chars):
                # core_span 보정
                prev_core = prev.meta.get("core_span", [prev.span_start, prev.span_end])
                curr_core = curr.meta.get("core_span", [curr.span_start, curr.span_end])
                try:
                    prev.meta["core_span"] = [int(prev_core[0]), int(curr_core[1])]
                except Exception:
                    prev.meta["core_span"] = [prev.span_start, curr.span_end]

                # span/text 확장
                prev.span_end = int(curr.span_end)
                try:
                    prev.text = _safe_slice(full_text, int(prev.span_start), int(prev.span_end))
                except Exception:
                    pass

                # curr 제거
                del chunks[i]
                merged += 1
                affected.append(prev_base)
                continue  # 같은 인덱스에 새 항목이 당겨졌으니 재검사
        i += 1

    return {"merged_count": merged, "affected_sections": affected}
