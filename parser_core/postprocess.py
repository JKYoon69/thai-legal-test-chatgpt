# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional
import re

from .schema import ParseResult, Node, Chunk
from .rules_th import normalize_text

def validate_tree(result: ParseResult) -> List[str]:
    """
    Very lightweight checks: level regressions, ultra-short leaves, etc.
    """
    issues: List[str] = []
    prev_level = 0
    for n in result.all_nodes:
        if n is result.root:
            continue
        if n.level < 0:
            issues.append(f"Invalid level at {n.label} {n.num}")
        if len(n.text.strip()) < 2 and n.label not in ("front_matter",):
            issues.append(f"Too short node: {n.label} {n.num}")
        prev_level = n.level
    return issues

def _is_leaf(n: Node) -> bool:
    # consider มาตรา / ข้อ as leaves (and anything without children at deepest level)
    if n.children:
        return False
    return n.label in ("มาตรา", "ข้อ")

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
    Heuristic: first line that contains พระราชบัญญัติ or ประมวลกฎหมาย
    """
    norm = normalize_text(text)
    for line in norm.splitlines()[:40]:
        if "พระราชบัญญัติ" in line or "ประมวลกฎหมาย" in line:
            return line.strip()
    return None

def _gather_leaves(root: Node) -> List[Node]:
    leaves: List[Node] = []
    stack = list(root.children)
    while stack:
        n = stack.pop(0)
        if _is_leaf(n):
            leaves.append(n)
        for c in n.children:
            stack.append(c)
    return leaves

def make_chunks(
    result: ParseResult,
    mode: str = "article±1",
    source_file: Optional[str] = None,
    law_name: Optional[str] = None,
) -> List[Chunk]:
    """
    Produce chunk objects ready for embedding/FTS ingestion.
    Modes:
      - article_only: each มาตรา/ข้อ as a chunk
      - article±1   : shallow merge with prev/next leaf
    """
    leaves = _gather_leaves(result.root)
    chunks: List[Chunk] = []

    for i, leaf in enumerate(leaves):
        merge_ids = [i]
        if mode == "article±1":
            if i > 0:
                merge_ids.insert(0, i - 1)
            if i + 1 < len(leaves):
                merge_ids.append(i + 1)

        # merge spans
        span_start = min(leaves[j].span_start for j in merge_ids)
        span_end = max(leaves[j].span_end for j in merge_ids)
        text = result.full_text[span_start:span_end]

        node_ids = [leaves[j].node_id for j in merge_ids]
        crumbs = _breadcrumbs(leaf, result)

        section_label = f"{leaf.label} {leaf.num}".strip()

        meta = {
            "mode": mode,
            "doc_type": result.doc_type,
            "law_name": law_name or "",
            "section_label": section_label,
            "source_file": source_file or "",
        }

        chunks.append(
            Chunk(
                text=text,
                span_start=span_start,
                span_end=span_end,
                node_ids=node_ids,
                breadcrumbs=crumbs,
                meta=meta,
            )
        )

    return chunks
