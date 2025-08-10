# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import uuid

def _nid() -> str:
    return uuid.uuid4().hex[:12]

@dataclass
class Node:
    level: int               # 0=root, 1..N
    label: Optional[str]     # e.g., ภาค / ลักษณะ / หมวด / มาตรา / ข้อ / front_matter
    num: Optional[str]       # e.g., 1, 1/1, ๑, etc. (already normalized upstream)
    span_start: int
    span_end: int
    text: str                # raw text slice for this node
    children: List["Node"] = field(default_factory=list)
    parent_id: Optional[str] = None
    node_id: str = field(default_factory=_nid)

@dataclass
class ParseResult:
    root: Node
    all_nodes: List[Node]
    node_map: Dict[str, Node]
    doc_type: str
    full_text: str

@dataclass
class Chunk:
    text: str
    span_start: int
    span_end: int
    node_ids: List[str]
    breadcrumbs: List[str]
    meta: Dict[str, str]
