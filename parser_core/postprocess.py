# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import re

from .schema import ParseResult, Node, Chunk
from .rules_th import normalize_text

MAX_NODE_CHARS = 10000  # 사실상 비활성

# ───────────────────────────── 레벨 랭킹 & 유틸 ───────────────────────────── #

_LEVEL_ORDER = {
    "root": 0,
    "front_matter": 1,
    "ภาค": 2,
    "ลักษณะ": 3,
    "หมวด": 4,
    "ส่วน": 5,
    "บท*": 6,  # 'บทบัญญัติทั่วไป', 'บทกำหนดโทษ' 등 prefix 취급
    "มาตรา": 7,
    "ข้อ": 7,
}

def _rank(label: Optional[str]) -> int:
    if not label:
        return 99
    if label.startswith("บท"):
        return _LEVEL_ORDER["บท*"]
    return _LEVEL_ORDER.get(label, 98)

def _is_article_leaf(n: Node) -> bool:
    return n.label in ("มาตรา", "ข้อ")

def _collect_article_leaves(root: Node) -> List[Node]:
    leaves: List[Node] = []
    stack = list(root.children)
    while stack:
        n = stack.pop(0)
        if _is_article_leaf(n):
            leaves.append(n)
        for c in n.children:
            stack.append(c)
    leaves.sort(key=lambda x: x.span_start)
    return leaves

def _breadcrumbs(n: Node, result: ParseResult) -> List[str]:
    crumbs = []
    cur = n
    while cur and cur.parent_id:
        crumbs.append(f"{cur.label or ''}{(' ' + cur.num) if cur.num else ''}".strip())
        cur = result.node_map.get(cur.parent_id)
        if cur and cur.label == "root":
            break
    return list(reversed(crumbs))

def _compute_union(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda x: x[0])
    merged: List[List[int]] = []
    for s, e in spans_sorted:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

def _subtract_intervals(outer: Tuple[int, int], covered: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    S, E = outer
    res: List[Tuple[int, int]] = []
    cur = S
    for s, e in covered:
        if e <= cur:
            continue
        if s > cur:
            res.append((cur, min(s, E)))
        cur = max(cur, e)
        if cur >= E:
            break
    if cur < E:
        res.append((cur, E))
    return [(s, e) for (s, e) in res if e > s]

# ───────────────────────────── 제목/연도 추출 ───────────────────────────── #

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
THAI_MONTHS = ["มกราคม","กุมภาพันธ์","มีนาคม","เมษายน","พฤษภาคม","มิถุนายน","กรกฎาคม","สิงหาคม","กันยายน","ตุลาคม","พฤศจิกายน","ธันวาคม"]

def _thai_to_arabic(s: str) -> str:
    return s.translate(_THAI_DIGITS)

def _is_promulgation_line(ln: str) -> bool:
    keys = [
        "ให้ไว้ ณ", "ให้ไว้  ณ", "ให้ไว้ ณ วันที่",
        "ประกาศ ณ", "ในพระปรมาภิไธย", "ราชกิจจานุเบกษา"
    ]
    if any(k in ln for k in keys):
        return True
    if "วันที่" in ln and any(m in ln for m in THAI_MONTHS):
        return True
    return False

_YEAR_PAT = re.compile(r"(พ\.?\s*ศ\.?|พศ)\s*([0-9]{4})")

def _extract_year_phrase(s: str) -> Optional[str]:
    s2 = _thai_to_arabic(s)
    m = _YEAR_PAT.search(s2)
    if not m:
        return None
    return f"พ.ศ. {m.group(2)}"

def guess_law_name(text: str) -> Optional[str]:
    """
    제목 + 부제 + 연도(พ.ศ.) 결합
    - 태국 숫자 → 아라비아 숫자 변환
    - 첫 표제행 + 다음 1~5행에서 부제/연도 탐색
    - 공포문/날짜 라인은 law_name에서 제외
    """
    norm = normalize_text(text)
    lines = [ln.strip(" \t-–—") for ln in norm.splitlines()[:160] if ln.strip()]
    base_idx = None
    for i, ln in enumerate(lines):
        if ("พระราชบัญญัติ" in ln) or ("ประมวลกฎหมาย" in ln):
            base_idx = i
            break
    if base_idx is None:
        return None

    title = lines[base_idx]
    subline = ""
    year_phrase = ""

    for j in range(base_idx + 1, min(base_idx + 6, len(lines))):
        ln = lines[j]
        if any(k in ln for k in ("ภาค", "ลักษณะ", "หมวด", "ส่วน", "บท", "มาตรา")):
            break
        if _is_promulgation_line(ln):
            continue
        if ("ให้ใช้" in ln) or ("ยาเสพติด" in ln) or ("ประมวลกฎหมาย" in ln):
            if len(ln) > len(subline):
                subline = ln
        if not subline:
            subline = ln
        yp = _extract_year_phrase(ln)
        if yp:
            year_phrase = yp

    parts = [title]
    if subline and subline not in title:
        parts.append(subline)
    if year_phrase and year_phrase not in title and year_phrase != subline:
        parts.append(year_phrase)

    cand = " ".join(parts)
    cand = _thai_to_arabic(cand)
    return " ".join(cand.split()).strip()

# ───────────────────────────── 검증 ───────────────────────────── #

def validate_tree(result: ParseResult) -> List[str]:
    issues: List[str] = []

    def _check(n: Node):
        for i, c in enumerate(n.children):
            if i + 1 < len(n.children):
                nxt = n.children[i + 1]
                if c.span_end > nxt.span_start:
                    issues.append(f"Span overlap at {c.label} {c.num} -> {nxt.label} {nxt.num}")
            if c.span_start < n.span_start or c.span_end > n.span_end:
                issues.append(f"child outside parent: {c.label} {c.num} in {n.label} {n.num}")
            _check(c)
    _check(result.root)

    for n in result.all_nodes:
        if n is result.root or n.label == "front_matter":
            continue
        if n.text is None or len(n.text.strip()) < 1:
            issues.append(f"Empty node: {n.label} {n.num}")
    return issues

# ───────────────────────────── 트리 수복(Reparent & Span-Repair) ───────────────────────────── #

def repair_tree(result: ParseResult) -> Dict[str, Any]:
    """
    부모-자식 경계 불일치 최소화:
      1) bottom-up으로 부모 span을 자식 union에 맞게 확장(축소는 보수적)
      2) 여전히 벗어나는 자식은 root로 최소 reparent
    무손실: coverage에는 영향 없음.
    """
    full = result.full_text
    diag: Dict[str, Any] = {
        "adjusted_parents": 0,
        "expanded_total_chars": 0,
        "shrunk_total_chars": 0,
    }

    nodes_by_depth: Dict[int, List[Node]] = {}
    max_depth = 0
    for n in result.all_nodes:
        nodes_by_depth.setdefault(n.level, []).append(n)
        if n.level > max_depth:
            max_depth = n.level

    for depth in range(max_depth - 1, -1, -1):
        for p in nodes_by_depth.get(depth, []):
            if not p.children:
                if p.span_start is not None and p.span_end is not None:
                    p.text = full[p.span_start:p.span_end]
                continue
            child_spans = [(c.span_start, c.span_end) for c in p.children if c.span_start is not None and c.span_end is not None]
            if not child_spans:
                continue
            new_start = min(s for s, _ in child_spans)
            new_end = max(e for _, e in child_spans)
            if p.span_start is None or p.span_end is None:
                p.span_start, p.span_end = new_start, new_end
                p.text = full[p.span_start:p.span_end]
                diag["adjusted_parents"] += 1
                continue

            old_s, old_e = p.span_start, p.span_end
            changed = False
            if new_start < old_s:
                diag["expanded_total_chars"] += (old_s - new_start)
                old_s = new_start
                changed = True
            if new_end > old_e:
                diag["expanded_total_chars"] += (new_end - old_e)
                old_e = new_end
                changed = True

            if changed:
                p.span_start, p.span_end = old_s, old_e
                p.text = full[p.span_start:p.span_end]
                diag["adjusted_parents"] += 1

    root = result.root
    for p in result.all_nodes:
        for c in list(p.children):
            if c.span_start < p.span_start or c.span_end > p.span_end:
                try:
                    p.children.remove(c)
                except ValueError:
                    pass
                c.parent_id = root.node_id
                root.children.append(c)
                diag["adjusted_parents"] += 1

    result.node_map = {n.node_id: n for n in result.all_nodes}
    return diag

# ───────────────────────────── 시리즈/경로 유틸 ───────────────────────────── #

def _article_series_index(leaves: List[Node]) -> Dict[str, int]:
    def main_int(num: Optional[str]) -> int:
        if not num:
            return 10**9
        try:
            return int(str(num).split("/")[0])
        except Exception:
            return 10**9
    series = 1
    prev = -1
    mapping: Dict[str, int] = {}
    for lf in leaves:
        cur = main_int(lf.num)
        if prev != -1 and cur < prev:
            series += 1
        mapping[lf.node_id] = series
        prev = cur
    return mapping

def _find_path_at_offset(result: ParseResult, offset: int) -> List[str]:
    path: List[str] = []
    def descend(n: Node):
        nonlocal path
        for c in n.children:
            if c.span_start <= offset < c.span_end:
                label = f"{c.label or ''}{(' ' + c.num) if c.num else ''}".strip()
                if label:
                    path.append(label)
                descend(c)
                break
    descend(result.root)
    return path

def _retrieval_weight_for(type_: str) -> float:
    return {
        "article": 1.0,
        "appendix": 0.8,
        "headnote": 0.5,
        "front_matter": 0.3,
        "orphan_gap": 0.2,
    }.get(type_, 0.5)

# ───────────────────────────── 청크 생성 ───────────────────────────── #

def make_chunks(
    result: ParseResult,
    mode: str = "article_only",
    source_file: Optional[str] = None,
    law_name: Optional[str] = None,
    *,
    include_front_matter: bool = True,
    include_headnotes: bool = True,
    include_gap_fallback: bool = True,
    allowed_headnote_levels: List[str] = ("ภาค","ลักษณะ","หมวด","ส่วน","บท"),
    min_headnote_len: int = 24,
    min_gap_len: int = 24,
    strict_lossless: bool = False,
    # 롱 조문 보조분할 + tail 병합
    split_long_articles: bool = False,
    split_threshold_chars: int = 1800,
    tail_merge_min_chars: int = 200,
    # NEW: 문맥 오버랩 + 소프트 컷
    overlap_chars: int = 200,
    soft_cut: bool = True,
) -> Tuple[List[Chunk], Dict[str, Any]]:
    """
    Strict:
      - headnote/gap 길이 필터 비활성
      - post-fill로 남은 간극 orphan_gap 보강
    + 롱 조문 분할 + tail 병합
    + (NEW) 파트 간 오버랩(문맥) 부여, 컷 위치 휴리스틱
    """
    text = result.full_text
    total_len = len(text)
    chunks: List[Chunk] = []
    diag: Dict[str, Any] = {
        "headnote_candidates": 0,
        "headnote_after_filter": 0,
        "final_headnote_count": 0,
        "removals": {
            "removed_short": 0,
            "removed_same_span_dedupe": 0,
            "removed_overlap_clip_to_zero": 0,
            "removed_empty_text": 0,
            "removed_invalid_span": 0,
        },
        "strict_post_fill": {"enabled": bool(strict_lossless), "filled_gaps": 0, "total_chars": 0, "spans": []},
        "allowed_headnote_levels_effective": list(allowed_headnote_levels) + ["บท* (prefix)"],
        "split": {
            "enabled": bool(split_long_articles),
            "threshold": int(split_threshold_chars),
            "series_total_counts": {},
        },
        "tail_merge": {
            "min_chars": int(tail_merge_min_chars),
            "merged_count": 0,
            "affected_sections": [],
        },
        "overlap": {
            "enabled": overlap_chars > 0,
            "overlap_chars": int(overlap_chars),
            "example_core_vs_expanded": [],
        },
    }

    # 1) article leaves
    leaves = _collect_article_leaves(result.root)
    series_map = _article_series_index(leaves)

    # 컷 휴리스틱
    _re_list_item = re.compile(r"(?m)^\s*(?:[\(\[]?\s*[0-9๑-๙]+[\)\.]|[-•])\s+")
    _re_soft_punct = re.compile(r"[;:ฯ]")

    def _pick_soft_cut(p: int, e: int, q_target: int) -> int:
        """
        soft_cut: \n\n > 주변 \n(±120) > 근처 구두점/두 칸 공백 > fallback
        """
        window = min(120, e - p)
        # 1) 단락 경계
        cut = text.rfind("\n\n", p + int(0.6 * (q_target - p)), min(q_target + window, e))
        if cut != -1 and cut - p >= 200:
            return cut
        # 2) 근처 줄바꿈
        left = text.rfind("\n", max(p, q_target - window), q_target + 1)
        right = text.find("\n", q_target, min(e, q_target + window))
        cand = -1
        if right != -1:
            cand = right
        if left != -1 and (cand == -1 or (q_target - left) <= (cand - q_target)):
            cand = left
        if cand != -1 and cand - p >= 200:
            return cand
        # 3) 구두점/두 칸 공백
        back = text.rfind("  ", max(p, q_target - 200), q_target + 1)
        if back != -1 and back - p >= 150:
            return back
        m = None
        for m_ in _re_soft_punct.finditer(text, max(p, q_target - 200), min(e, q_target + 1)):
            m = m_
        if m and (m.end() - p) >= 150:
            return m.end()
        # 4) 백업
        return q_target

    def _split_paragraph_smart(s: int, e: int) -> List[Tuple[int, int]]:
        """핵심(core) 스팬 리스트 반환"""
        if not split_long_articles or (e - s) <= split_threshold_chars:
            return [(s, e)]
        spans: List[Tuple[int, int]] = []
        p = s
        while p < e:
            q_target = min(p + split_threshold_chars, e)
            cut = _pick_soft_cut(p, e, q_target) if soft_cut else q_target
            if cut <= p:
                cut = min(p + split_threshold_chars, e)
            spans.append((p, cut))
            p = cut
        return spans

    def _apply_tail_merge(spans: List[Tuple[int, int]], section_uid: str) -> List[Tuple[int, int]]:
        if len(spans) <= 1:
            return spans
        last_s, last_e = spans[-1]
        if (last_e - last_s) < tail_merge_min_chars:
            prev_s, prev_e = spans[-2]
            merged = spans[:-2] + [(prev_s, last_e)]
            diag["tail_merge"]["merged_count"] += 1
            if len(diag["tail_merge"]["affected_sections"]) < 10:
                diag["tail_merge"]["affected_sections"].append(section_uid)
            return merged
        return spans

    article_index = 0
    for leaf in leaves:
        article_index += 1
        section_label = f"{leaf.label} {leaf.num}".strip()
        crumbs = _breadcrumbs(leaf, result)
        section_uid = "/".join(crumbs) if crumbs else section_label

        core_spans = _split_paragraph_smart(leaf.span_start, leaf.span_end)
        core_spans = _apply_tail_merge(core_spans, section_uid)
        series_total = len(core_spans)
        diag["split"]["series_total_counts"][str(series_total)] = diag["split"]["series_total_counts"].get(str(series_total), 0) + 1

        prev_exp_end = None
        for k, (cs, ce) in enumerate(core_spans, 1):
            # 오버랩 확장
            s = max(leaf.span_start, cs - max(0, overlap_chars))
            e = min(leaf.span_end, ce + max(0, overlap_chars))
            if prev_exp_end is not None and s < prev_exp_end:
                s = prev_exp_end  # 앞 파트와 겹치면 앞의 끝에 맞춤
            if e <= s:
                continue

            label = section_label + (f" (part {k})" if series_total > 1 else "")
            meta = {
                "type": "article",
                "mode": mode,
                "doc_type": result.doc_type,
                "law_name": law_name or "",
                "section_label": label,
                "section_uid": section_uid,
                "source_file": source_file or "",
                "article_index_global": str(article_index),
                "series": str(series_map.get(leaf.node_id, 1)),
                "retrieval_weight": str(_retrieval_weight_for("article")),
                "series_index": str(k),
                "series_total": str(series_total),
                # NEW
                "core_span": [int(cs), int(ce)],
                "overlap_chars": str(overlap_chars),
            }
            chunks.append(
                Chunk(
                    text=text[s:e],
                    span_start=s,
                    span_end=e,
                    node_ids=[leaf.node_id],
                    breadcrumbs=crumbs,
                    meta=meta,
                )
            )
            prev_exp_end = e

            # 진단 예시 3건만 수집
            if len(diag["overlap"]["example_core_vs_expanded"]) < 3:
                diag["overlap"]["example_core_vs_expanded"].append(
                    {"core": [int(cs), int(ce)], "expanded": [int(s), int(e)], "section_uid": section_uid}
                )

    # 2) front_matter
    if include_front_matter:
        for n in result.root.children:
            if n.label == "front_matter" and (n.span_end - n.span_start) > 0:
                frag = text[n.span_start:n.span_end]
                if strict_lossless or frag.strip():
                    crumbs = ["front_matter"]
                    meta = {
                        "type": "front_matter",
                        "mode": mode,
                        "doc_type": result.doc_type,
                        "law_name": law_name or "",
                        "section_label": "front_matter",
                        "section_uid": "front_matter",
                        "source_file": source_file or "",
                        "retrieval_weight": str(_retrieval_weight_for("front_matter")),
                        "series_index": "1",
                        "series_total": "1",
                    }
                    chunks.append(
                        Chunk(
                            text=frag,
                            span_start=n.span_start,
                            span_end=n.span_end,
                            node_ids=[n.node_id],
                            breadcrumbs=crumbs,
                            meta=meta,
                        )
                    )
                break

    # 3) headnotes (Strict면 길이 필터 해제)
    if include_headnotes:
        allowed = set(allowed_headnote_levels)

        def is_allowed_label(lbl: Optional[str]) -> bool:
            if not lbl:
                return False
            return (lbl in allowed) or lbl.startswith("บท")

        def walk(parent: Node):
            nonlocal diag
            if is_allowed_label(parent.label):
                child_spans = [(c.span_start, c.span_end) for c in parent.children]
                covered = _compute_union(child_spans)
                leftovers = _subtract_intervals((parent.span_start, parent.span_end), covered)
                for (s, e) in leftovers:
                    frag = text[s:e]
                    diag["headnote_candidates"] += 1
                    keep = (e - s) > 0 if strict_lossless else (len(frag.strip()) >= min_headnote_len)
                    if keep:
                        crumbs = _breadcrumbs(parent, result)
                        section_label = f"{parent.label} {parent.num}".strip() if parent.num else parent.label or "headnote"
                        section_uid = ("/".join(crumbs) if crumbs else section_label) + " — headnote"
                        meta = {
                            "type": "headnote",
                            "mode": mode,
                            "doc_type": result.doc_type,
                            "law_name": law_name or "",
                            "section_label": f"{section_label} — headnote",
                            "section_uid": section_uid,
                            "source_file": source_file or "",
                            "retrieval_weight": str(_retrieval_weight_for("headnote")),
                            "series_index": "1",
                            "series_total": "1",
                        }
                        chunks.append(
                            Chunk(
                                text=frag,
                                span_start=s,
                                span_end=e,
                                node_ids=[parent.node_id],
                                breadcrumbs=crumbs,
                                meta=meta,
                            )
                        )
                        diag["headnote_after_filter"] += 1
                    else:
                        diag["removals"]["removed_short"] += 1
            for c in parent.children:
                walk(c)
        walk(result.root)

    # 4) 1차 gap-sweeper
    if include_gap_fallback:
        ivs = _compute_union([(c.span_start, c.span_end) for c in chunks])
        gaps: List[Tuple[int, int]] = []
        prev = 0
        for s, e in ivs:
            if s > prev:
                gaps.append((prev, s))
            prev = e
        if prev < total_len:
            gaps.append((prev, total_len))

        for idx, (s, e) in enumerate(gaps, 1):
            if e <= s:
                diag["removals"]["removed_invalid_span"] += 1
                continue
            frag = text[s:e]
            keep = (e - s) > 0 if strict_lossless else (len(frag.strip()) >= min_gap_len)
            if keep:
                crumbs = _find_path_at_offset(result, s)
                meta = {
                    "type": "orphan_gap",
                    "mode": mode,
                    "doc_type": result.doc_type,
                    "law_name": law_name or "",
                    "section_label": f"orphan_gap #{idx}",
                    "section_uid": f"gap[{s}:{e}]",
                    "source_file": source_file or "",
                    "retrieval_weight": str(_retrieval_weight_for("orphan_gap")),
                    "series_index": "1",
                    "series_total": "1",
                }
                chunks.append(
                    Chunk(
                        text=frag,
                        span_start=s,
                        span_end=e,
                        node_ids=[],
                        breadcrumbs=crumbs,
                        meta=meta,
                    )
                )
            else:
                diag["removals"]["removed_short"] += 1

    # ── 정리: 정렬 → 동기화 → 동일 스팬 dedupe → 연속 겹침 클립 ── #
    chunks.sort(key=lambda c: (c.span_start, c.span_end))

    cleaned: List[Chunk] = []
    for c in chunks:
        s, e = c.span_start, c.span_end
        if s is None or e is None or e <= s:
            diag["removals"]["removed_invalid_span"] += 1
            continue
        c.text = text[s:e]
        if not strict_lossless and c.text.strip() == "":
            diag["removals"]["removed_empty_text"] += 1
            continue
        cleaned.append(c)
    chunks = cleaned

    # 동일 스팬 dedupe
    seen_span = set()
    dedup1: List[Chunk] = []
    for c in chunks:
        key = (c.span_start, c.span_end)
        if key in seen_span:
            diag["removals"]["removed_same_span_dedupe"] += 1
            continue
        seen_span.add(key)
        dedup1.append(c)
    chunks = dedup1

    # 연속 겹침 → 뒤 청크 전방 클립
    final: List[Chunk] = []
    last_end = -1
    for c in chunks:
        s, e = c.span_start, c.span_end
        if last_end > -1 and s < last_end:
            s = last_end
            if e - s <= 0:
                diag["removals"]["removed_overlap_clip_to_zero"] += 1
                continue

        if not strict_lossless:
            if c.meta.get("type") in ("headnote", "orphan_gap", "front_matter"):
                thr = min_headnote_len if c.meta.get("type") == "headnote" else min_gap_len
                if (e - s) < thr:
                    diag["removals"]["removed_short"] += 1
                    continue

        c.span_start, c.span_end = s, e
        c.text = text[s:e]
        final.append(c)
        last_end = e

    # headnote 최종 개수
    diag["final_headnote_count"] = sum(1 for c in final if c.meta.get("type") == "headnote")

    # Strict post-fill
    if strict_lossless:
        ivs2 = _compute_union([(c.span_start, c.span_end) for c in final])
        post_gaps: List[Tuple[int, int]] = []
        prev = 0
        for s, e in ivs2:
            if s > prev:
                post_gaps.append((prev, s))
            prev = e
        if prev < total_len:
            post_gaps.append((prev, total_len))

        for (s, e) in post_gaps:
            if e <= s:
                diag["removals"]["removed_invalid_span"] += 1
                continue
            frag = text[s:e]
            crumbs = _find_path_at_offset(result, s)
            meta = {
                "type": "orphan_gap",
                "mode": mode,
                "doc_type": result.doc_type,
                "law_name": law_name or "",
                "section_label": f"strict_fill_gap[{s}:{e}]",
                "section_uid": f"strict_gap[{s}:{e}]",
                "source_file": source_file or "",
                "retrieval_weight": str(_retrieval_weight_for("orphan_gap")),
                "strict_fill": "1",
                "series_index": "1",
                "series_total": "1",
            }
            final.append(
                Chunk(
                    text=frag,
                    span_start=s,
                    span_end=e,
                    node_ids=[],
                    breadcrumbs=crumbs,
                    meta=meta,
                )
            )
            diag["strict_post_fill"]["filled_gaps"] += 1
            diag["strict_post_fill"]["total_chars"] += (e - s)
            if len(diag["strict_post_fill"]["spans"]) < 5:
                diag["strict_post_fill"]["spans"].append([s, e])

        final.sort(key=lambda c: (c.span_start, c.span_end))

    return final, diag

# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple

# ... (기존 코드들 그대로 두세요)

PART_RE = re.compile(r"^(?P<base>.+?)\s*\(part\s*(?P<idx>\d+)\)\s*$", re.IGNORECASE)

def _part_info(label: str) -> Tuple[str, int | None]:
    """
    'มาตรา 119 (part 2)' -> ('มาตรา 119', 2)
    'มาตรา 119' -> ('มาตรา 119', None)
    """
    if not isinstance(label, str):
        return ("", None)
    m = PART_RE.match(label.strip())
    if not m:
        return (label.strip(), None)
    return (m.group("base").strip(), int(m.group("idx")))

def merge_small_trailing_parts(
    chunks: List["Chunk"],
    *,
    full_text: str,
    max_tail_chars: int = 200
) -> Dict[str, Any]:
    """
    분할 시리즈의 마지막(part N) 꼬리가 너무 짧으면 앞 청크로 병합.
    - 같은 series(base)가 맞고, 현재가 마지막(part N), 길이<=max_tail_chars일 때만
    - 병합: prev.span_end 및 prev.core_span[1]을 curr의 끝까지 확장하고 curr 제거
    반환: {'merged_count':int, 'affected_sections':[...]}
    """
    if not chunks:
        return {"merged_count": 0, "affected_sections": []}

    # span 순서로 가정 (make_chunks가 이미 정렬)
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

            # 같은 시리즈의 연속 파트인지 확인
            is_series = (prev_base and curr_base and prev_base == curr_base and
                         prev_idx is not None and curr_idx is not None and curr_idx == prev_idx + 1)

            curr_len = int(curr.span_end) - int(curr.span_start)

            if is_series and curr_len <= int(max_tail_chars):
                # 병합
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
                    prev.text = full_text[int(prev.span_start):int(prev.span_end)]
                except Exception:
                    # 안전 폴백: 텍스트는 그대로 두되 span만 확장
                    pass

                # 현재 조각 제거
                del chunks[i]
                merged += 1
                affected.append(prev_base)
                # i는 그대로(다음 항목이 앞으로 땡겨졌으니 같은 i 인덱스를 다시 검사)
                continue
        i += 1

    return {"merged_count": merged, "affected_sections": affected}
