# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple
import re

from .rules_th import normalize_text, RE_HEADER, label_level
from .schema import Node, ParseResult

def detect_doc_type(text: str) -> str:
    """
    Super-light heuristic: could be extended (e.g., กฎกระทรวง, พระราชบัญญัติ, ประกาศ)
    """
    t = text[:400]
    if "พระราชบัญญัติ" in t:
        return "act"
    if "ประมวลกฎหมาย" in t:
        return "code"
    return "unknown"

def _scan_headers(text: str) -> List[Tuple[str, Optional[str], int, int]]:
    """
    Returns list of (label, num, span_start, span_end_of_header_line)
    """
    headers: List[Tuple[str, Optional[str], int, int]] = []
    for label, pat in RE_HEADER.items():
        for m in pat.finditer(text):
            num = m.groupdict().get("num")
            s, e = m.span()
            headers.append((label, num, s, e))
    headers.sort(key=lambda x: x[2])  # by start
    return headers

def parse_document(text: str, doc_type: str = "unknown") -> ParseResult:
    """
    Build a hierarchy tree using a simple stack by LEVEL order.
    front_matter is created for any text before the first header.
    Each node's span_end is resolved by next header start, except the last → EOF.
    """
    norm = normalize_text(text)
    headers = _scan_headers(norm)

    # root
    root = Node(level=0, label="root", num=None, span_start=0, span_end=len(norm), text="")
    all_nodes = [root]
    node_map = {root.node_id: root}

    # create front_matter if non-empty prefix
    first_start = headers[0][2] if headers else 0
    if first_start > 0:
        fm = Node(
            level=1, label="front_matter", num=None,
            span_start=0, span_end=first_start, text=norm[:first_start], parent_id=root.node_id
        )
        root.children.append(fm)
        all_nodes.append(fm)
        node_map[fm.node_id] = fm

    # stack starts from root
    stack: List[Node] = [root]

    # create nodes for each header
    for idx, (label, num, h_start, h_header_end) in enumerate(headers):
        lvl = label_level(label)
        # close previous open node(s) if deeper/equal
        while stack and stack[-1].level >= lvl:
            stack.pop()
        parent = stack[-1] if stack else root

        # determine provisional end = next header start, else EOF; set later after we know next start
        next_start = headers[idx + 1][2] if idx + 1 < len(headers) else len(norm)

        node_text = norm[h_start:next_start]

        node = Node(
            level=lvl, label=label, num=num, span_start=h_start, span_end=next_start,
            text=node_text, parent_id=parent.node_id
        )
        parent.children.append(node)
        all_nodes.append(node)
        node_map[node.node_id] = node

        # push
        stack.append(node)

    return ParseResult(root=root, all_nodes=all_nodes, node_map=node_map, doc_type=doc_type, full_text=norm)
