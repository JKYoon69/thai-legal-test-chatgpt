# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional, Tuple

from .schema import ParseResult, Node, Chunk
from .rules_th import normalize_text

# limits for safety; we keep article-only, but avoid absurd outliers
MAX_NODE_CHARS = 4000     # 조문 텍스트가 이보다 길면 조문 내부에서 2~3개로만 안전 분할
MIN_CHUNK_CHARS = 500     # 너무 짧은 경우 이웃과 합쳐 최소 길이 보장

def validate_tree(result: ParseResult) -> List[str]:
    issues: List[str] = []
    # 1) sibling span order
    def _check(n: Node):
        for i, c in enumerate(n.children):
            if i + 1 < len(n.children):
                nxt = n.children[i + 1]
                if c.span_end > nxt.span_start:
                    issues.append(f"Span overlap at {c.label} {c.num} -> {nxt.label} {nxt.num}")
            _check(c)
    _check(result.root)

    # 2) empty/ultra-short nodes (except front_matter)
    for n in result.all_nodes:
        if n is result.root or n.label == "front_matter":
            continue
        if len(n.text.strip()) < 2:
            issues.append(f"Empty node: {n.label} {n.num}")

    return issues

def _is_leaf(n: Node) -> bool:
    # treat มาตรา / ข้อ as leaves; if deeper levels exist, still consider article-level leaves
    if n.children:
        return False
    return n.label in ("มาตรา", "ข้อ")

def _gather_leaves(root: Node) -> List[Node]:
    leaves: List[Node] = []
    stack = list(root.children)
    while stack:
        n = stack.pop(0)
        if _is_leaf(n):
            leaves.append(n)
        for c in n.children:
            stack.append(c)
    # SAFETY: sort by document order
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

def guess_law_name(text: str) -> Optional[str]:
    """
    Slightly stronger: get 1~2 consecutive lines near the top containing title tokens.
    """
    norm = normalize_text(text)
    lines = [ln.strip() for ln in norm.splitlines()[:60] if ln.strip()]
    title = None
    for i, ln in enumerate(lines):
        if ("พระราชบัญญัติ" in ln) or ("ประมวลกฎหมาย" in ln):
            title = ln
            # if next line looks like subtitle/series, append
            if i + 1 < len(lines) and len(lines[i + 1]) <= 60 and ("ภาค" in lines[i+1] or "ลักษณะ" in lines[i+1] or "พ.ศ." in lines[i+1] or "ฉบับ" in lines[i+1]):
                title = f"{title} / {lines[i+1]}"
            break
    return title

def _split_long_article(text: str, start: int, limit: int) -> List[Tuple[int, int]]:
    """
    If article text is too long, split on paragraph breaks to keep <= limit.
    Returns list of (s, e) offsets relative to full document.
    """
    # try to split by double newline boundaries
    relative = text
    if len(relative) <= limit:
        return [(start, start + len(relative))]
    parts: List[Tuple[int, int]] = []
    s = 0
    while s < len(relative):
        e = min(s + limit, len(relative))
        # move e to nearest paragraph boundary if possible
        candidate = relative.rfind("\n\n", s, e)
        if candidate != -1 and candidate > s + MIN_CHUNK_CHARS:
            e = candidate + 2  # include the boundary
        parts.append((start + s, start + e))
        s = e
    return parts

def make_chunks(
    result: ParseResult,
    mode: str = "article_only",
    source_file: Optional[str] = None,
    law_name: Optional[str] = None,
) -> List[Chunk]:
    """
    Article-only chunking with safety:
      - leaves sorted by span_start
      - overly long article split by paragraph boundaries (kept within the same article)
      - very short fragments merged into neighbors when possible
    """
    leaves = _gather_leaves(result.root)
    chunks: List[Chunk] = []

    # First pass: create article chunks (with split if needed)
    tmp: List[Chunk] = []
    for leaf in leaves:
        section_label = f"{leaf.label} {leaf.num}".strip()
        meta = {
            "mode": "article_only",
            "doc_type": result.doc_type,
            "law_name": law_name or "",
            "section_label": section_label,
            "source_file": source_file or "",
        }
        spans = _split_long_article(
            text=result.full_text[leaf.span_start:leaf.span_end],
            start=leaf.span_start,
            limit=MAX_NODE_CHARS,
        )
        # create chunks per span
        for k, (s, e) in enumerate(spans, 1):
            part_suffix = f" (part {k})" if len(spans) > 1 else ""
            tmp.append(
                Chunk(
                    text=result.full_text[s:e],
                    span_start=s,
                    span_end=e,
                    node_ids=[leaf.node_id],
                    breadcrumbs=_breadcrumbs(leaf, result),
                    meta={**meta, "section_label": section_label + part_suffix},
                )
            )

    # Second pass: merge too-short chunks to neighbors
    i = 0
    while i < len(tmp):
        cur = tmp[i]
        if len(cur.text) < MIN_CHUNK_CHARS and i + 1 < len(tmp):
            nxt = tmp[i + 1]
            # merge into next
            merged = Chunk(
                text=cur.text + nxt.text,
                span_start=cur.span_start,
                span_end=nxt.span_end,
                node_ids=list({*cur.node_ids, *nxt.node_ids}),
                breadcrumbs=cur.breadcrumbs,
                meta=cur.meta,
            )
            tmp[i + 1] = merged
            i += 2
            continue
        chunks.append(cur)
        i += 1

    # Ensure final order
    chunks.sort(key=lambda c: c.span_start)
    return chunks
