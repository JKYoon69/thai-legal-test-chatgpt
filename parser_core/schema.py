from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List, Dict


class Span(BaseModel):
    start: int
    end: int


class Node(BaseModel):
    node_id: str
    level: str            # 문서/ภาค/หมวด/มาตรา/ข้อ/บัญชีท้าย 등
    label: str            # "มาตรา 12" 같은 원문 라벨
    num: Optional[str] = None  # "12" (태국숫자→아라비아 변환된 문자열)
    span: Span
    text: str
    breadcrumbs: List[str] = []
    children: List["Node"] = []


class ParseResult(BaseModel):
    doc_type: str
    nodes: List[Node]
    root_nodes: List[Node]
    stats: Dict[str, int] = {}


class Issue(BaseModel):
    level: str   # info/warn/error
    message: str


class Chunk(BaseModel):
    chunk_id: str
    node_ids: List[str]       # 포함된 노드 (보통 조문 1개 또는 ±1 병합)
    text: str
    breadcrumbs: List[str]
    span: Span
    meta: Dict[str, str] = {}
