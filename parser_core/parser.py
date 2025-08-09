from . import rules_th
from .schema import Node
from typing import List

def detect_doc_type(text: str) -> str:
    # 간단한 유형 감지
    if "ประมวล" in text:
        return "code"
    elif "พระราชบัญญัติ" in text:
        return "act"
    elif "พระราชกฤษฎีกา" in text:
        return "royal_decree"
    return "unknown"

def parse_document(text: str, forced_type: str = None) -> List[Node]:
    doc_type = forced_type or detect_doc_type(text)
    # TODO: 규칙 파서 로직 작성
    return []
