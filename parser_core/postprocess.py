# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple, Dict

from .schema import ParseResult, Node, Chunk
from .rules_th import normalize_text

# 조문 분할 상한(사실상 비활성)
MAX_NODE_CHARS = 10000

# ───────────────────────────── 검증 ───────────────────────────── #

def validate_tree(result: ParseResult) -> List[str]:
    issues: List[str] = []

    def _check(n: Node):
        for i, c in enumerate(n.children):
            if i + 1 < len(n.children):
                nxt = n.children[i + 1]
                if c.span_end > nxt.span_start:
                    issues.append(f"Span overlap at {c.label} {c.num} -> {nxt.label} {nxt.num}")
            _check(c)
    _check(result.root)

    for n in result.all_nodes:
        if n is result.root or n.label == "front_matter":
            continue
        if n.text is None or len(n.text.strip()) < 1:
            issues.append(f"Empty node: {n.label} {n.num}")
    return issues

# ───────────────────────────── 유틸 ───────────────────────────── #

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

def guess_law_name(text: str) -> Optional[str]:
    norm = normalize_text(text)
    lines = [ln.strip() for ln in norm.splitlines()[:60] if ln.strip()]
    title = None
    for i, ln in enumerate(lines):
        if ("พระราชบัญญัติ" in ln) or ("ประมวลกฎหมาย" in ln):
            title = ln
            if i + 1 < len(lines) and len(lines[i + 1]) <= 60 and (
                "ภาค" in lines[i+1] or "ลักษณะ" in lines[i+1] or "พ.ศ." in lines[i+1] or "ฉบับ" in lines[i+1] or "ประมวลกฎหมาย" in lines[i+1]
            ):
                title = f"{title} / {lines[i+1]}"
            break
    return title

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
) -> List[Chunk]:
    """
    조문 1:1 + front_matter/headnote/gap 보강(옵션) + 중복/겹침 억제
    """
    text = result.full_text
    total_len = len(text)
    chunks: List[Chunk] = []

    # 1) article leaves
    leaves = _collect_article_leaves(result.root)
    series_map = _article_series_index(leaves)

    article_index = 0
    for leaf in leaves:
        article_index += 1
        section_label = f"{leaf.label} {leaf.num}".strip()
        crumbs = _breadcrumbs(leaf, result)
        section_uid = "/".join(crumbs) if crumbs else section_label

        spans = [(leaf.span_start, leaf.span_end)]
        new_spans: List[Tuple[int, int]] = []
        for s, e in spans:
            frag = text[s:e]
            if len(frag) <= MAX_NODE_CHARS:
                new_spans.append((s, e))
            else:
                p = s
                while p < e:
                    q = min(p + MAX_NODE_CHARS, e)
                    cut = text.rfind("\n\n", p, q)
                    if cut != -1 and (cut - p) > 200:
                        q = cut + 2
                    new_spans.append((p, q))
                    p = q

        for k, (s, e) in enumerate(new_spans, 1):
            label = section_label + (f" (part {k})" if len(new_spans) > 1 else "")
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

    # 2) front_matter
    if include_front_matter:
        for n in result.root.children:
            if n.label == "front_matter" and (n.span_end - n.span_start) > 0:
                frag = text[n.span_start:n.span_end]
                if frag.strip():
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

    # 3) headnotes (레벨 제한 + 최소 길이)
    if include_headnotes:
        allowed = set(allowed_headnote_levels)
        def walk(parent: Node):
            if parent.label in allowed:
                child_spans = [(c.span_start, c.span_end) for c in parent.children]
                covered = _compute_union(child_spans)
                leftovers = _subtract_intervals((parent.span_start, parent.span_end), covered)
                for (s, e) in leftovers:
                    frag = text[s:e]
                    if len(frag.strip()) >= min_headnote_len:
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
            for c in parent.children:
                walk(c)
        walk(result.root)

    # 4) gap-sweeper (짧은 gap은 무시)
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
            frag = text[s:e]
            if len(frag.strip()) >= min_gap_len:
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

    # ── 정리: 정렬 → 동기화 → 동일구간 dedupe → 연속 겹침 클립 ── #
    chunks.sort(key=lambda c: (c.span_start, c.span_end))

    cleaned: List[Chunk] = []
    for c in chunks:
        s, e = c.span_start, c.span_end
        if s is None or e is None or e <= s:
            continue
        c.text = text[s:e]  # 항상 소스 기준 재생성
        if c.text.strip() == "":
            continue
        cleaned.append(c)
    chunks = cleaned

    seen_span = set()
    dedup1: List[Chunk] = []
    for c in chunks:
        key = (c.span_start, c.span_end)
        if key in seen_span:
            continue
        seen_span.add(key)
        dedup1.append(c)
    chunks = dedup1

    final: List[Chunk] = []
    last_end = -1
    for c in chunks:
        s, e = c.span_start, c.span_end
        if last_end > -1 and s < last_end:
            s = last_end  # 뒤 청크 전방 클립
        if e - s <= 0:
            continue
        if c.meta.get("type") in ("headnote", "orphan_gap", "front_matter"):
            if (e - s) < min( min_headnote_len if c.meta.get("type")=="headnote" else min_gap_len, 8):
                continue
        c.span_start, c.span_end = s, e
        c.text = text[s:e]
        final.append(c)
        last_end = e

    return final
