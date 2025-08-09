from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List, Dict

class Span(BaseModel):
    start: int
    end: int

class Node(BaseModel):
    node_id: str
    level: str            # 문서/ภาค/หมวด/มาตรา/ข้อ/appendix 등
    label: str            # "มาตรา 12" 같은 원문 라벨(숫자는 아라비아 표준화)
    num: Optional[str] = None
    span: Span
    # NOTE: 용량 줄이기 위해 text는 저장하지 않음 (chunk 생성 시 slice)
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
    node_ids: List[str]
    text: str
    breadcrumbs: List[str]
    span: Span
    meta: Dict[str, str] = {}
