# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple
import re

from .rules_th import normalize_text, RE_HEADER, label_level
from .schema import Node, ParseResult

def detect_doc_type(text: str) -> str:
    t = text[:600]
    if "ประมวลกฎหมาย" in t:
        return "code"
    if "พระราชบัญญัติ" in t:
        return "act"
    return "unknown"

def _scan_headers(text: str) -> List[Tuple[str, Optional[str], int, int]]:
    """
    Return sorted list of (label, num, header_start, header_line_end)
    """
    headers: List[Tuple[str, Optional[str], int, int]] = []
    for label, pat in RE_HEADER.items():
        for m in pat.finditer(text):
            num = m.groupdict().get("num")
            s, e = m.span()
            headers.append((label, num, s, e))
    headers.sort(key=lambda x: x[2])
    return headers

def parse_document(text: str, doc_type: str = "unknown") -> ParseResult:
    """
    Hardened stack-based hierarchy builder.
    - front_matter covers any prefix before 1st header
    - consecutive same-/higher-level headers close previous nodes
    - span_end = next header start (no overlaps)
    """
    norm = normalize_text(text)
    headers = _scan_headers(norm)

    root = Node(level=0, label="root", num=None, span_start=0, span_end=len(norm), text="")
    all_nodes: List[Node] = [root]
    node_map = {root.node_id: root}

    # front_matter
    first_start = headers[0][2] if headers else 0
    if first_start > 0:
        fm = Node(
            level=1, label="front_matter", num=None,
            span_start=0, span_end=first_start, text=norm[:first_start], parent_id=root.node_id
        )
        root.children.append(fm)
        all_nodes.append(fm)
        node_map[fm.node_id] = fm

    stack: List[Node] = [root]  # always non-empty

    # Build nodes
    for idx, (label, num, h_start, _h_end) in enumerate(headers):
        lvl = label_level(label)

        # close nodes at same or deeper level
        while stack and stack[-1].level >= lvl:
            stack.pop()
        parent = stack[-1] if stack else root

        next_start = headers[idx + 1][2] if idx + 1 < len(headers) else len(norm)
        # guard: if headers overlapped or malformed, clamp
        if next_start < h_start:
            next_start = len(norm)

        node_text = norm[h_start:next_start]

        node = Node(
            level=lvl, label=label, num=num, span_start=h_start, span_end=next_start,
            text=node_text, parent_id=parent.node_id
        )
        parent.children.append(node)
        all_nodes.append(node)
        node_map[node.node_id] = node

        stack.append(node)

    # Final sanity: enforce non-decreasing spans among siblings
    def _fix_spans(n: Node):
        for i, c in enumerate(n.children):
            if i + 1 < len(n.children):
                nxt = n.children[i + 1]
                if c.span_end > nxt.span_start:
                    c.span_end = nxt.span_start
            _fix_spans(c)

    _fix_spans(root)

    return ParseResult(root=root, all_nodes=all_nodes, node_map=node_map, doc_type=doc_type, full_text=norm)
