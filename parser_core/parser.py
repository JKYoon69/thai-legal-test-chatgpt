from __future__ import annotations
from typing import List, Tuple
import regex as re
from .schema import Node, Span, ParseResult
from . import rules_th as R


def detect_doc_type(text: str) -> str:
    t = text[:10000].translate(R.THAI2ARABIC)
    if "ประมวล" in t:
        return "code"
    if "พระราชบัญญัติประกอบรัฐธรรมนูญ" in t or "พระราชบัญญัติ" in t:
        return "act"
    if "พระราชกฤษฎีกา" in t:
        return "royal_decree"
    if "ระเบียบ" in t or "ประกาศ" in t or "คำสั่ง" in t:
        return "regulation"
    return "unknown"


def _find_all_markers(text: str, doc_type: str) -> List[Tuple[int, int, str, str]]:
    """
    줄 시작(anchor)에서만 레벨 토큰을 탐지하여 (start, end, level, label_with_num) 반환
    """
    levels = R.LEVELS.get(doc_type, R.LEVELS["unknown"])
    markers = []
    for lv in levels:
        pat = R.RE_HEADER.get(lv)
        if not pat:
            continue
        for m in pat.finditer(text):
            # 괄호 등으로 감싼 라인 제외: 예) "(มาตรา 12 ...)" 방지
            line_start = text.rfind("\n", 0, m.start()) + 1
            if line_start >= 1 and line_start < m.start():
                if text[line_start] in "([（":  # 다양한 괄호 앞머리
                    continue
            start, end = m.span()
            num = m.group("num")
            markers.append((start, end, lv, f"{lv} {num}"))

    # 부록류
    for m in R.RE_APPENDIX.finditer(text):
        start, end = m.span()
        markers.append((start, end, "appendix", text[start:end].strip()))

    markers.sort(key=lambda x: x[0])

    # 중복/너무 근접한 라벨 제거(실수 매칭 대비)
    deduped = []
    last_pos = -999999
    for m in markers:
        if m[0] - last_pos < 3:
            continue
        deduped.append(m)
        last_pos = m[0]
    return deduped


def parse_document(text: str, forced_type: str | None = None) -> ParseResult:
    norm = R.normalize_text(text)
    doc_type = forced_type or detect_doc_type(norm)

    markers = _find_all_markers(norm, doc_type)
    if not markers:
        # 헤더가 전혀 없으면 문서 전체를 루트 1개로
        root = Node(
            node_id="document-1",
            level="document",
            label="document",
            num=None,
            span=Span(start=0, end=len(norm)),
            breadcrumbs=["document"],
            children=[]
        )
        return ParseResult(doc_type=doc_type, nodes=[root], root_nodes=[root],
                           stats={"node_count":1, "leaf_count":1})

    # 경계 계산
    boundaries = []
    for i, (s, e, lv, label) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(norm)
        boundaries.append((s, e, next_start, lv, label))

    nodes: List[Node] = []
    root_nodes: List[Node] = []

    level_order = {lv: i for i, lv in enumerate(R.LEVELS.get(doc_type, R.LEVELS["unknown"]))}
    stack: List[Node] = []

    def _push(node: Node):
        if not stack:
            root_nodes.append(node)
        else:
            stack[-1].children.append(node)
        stack.append(node)

    for s, e, nxt, lv, label in boundaries:
        lvl_rank = 999 if lv == "appendix" else level_order.get(lv, 998)

        while stack:
            top_rank = 999 if stack[-1].level == "appendix" else level_order.get(stack[-1].level, 998)
            if top_rank <= lvl_rank:
                break
            stack.pop()

        m = re.search(r"(\d{1,4}(?:/\d{1,3})?)", label)
        num = m.group(1) if m else None

        node = Node(
            node_id=f"{lv}-{num or len(nodes)+1}",
            level=lv,
            label=label,
            num=num,
            span=Span(start=s, end=nxt),
            breadcrumbs=[],
            children=[],
        )
        _push(node)
        nodes.append(node)

    # breadcrumbs 채우기
    def fill_breadcrumbs(node: Node, trail: List[str]):
        node.breadcrumbs = trail + [f"{node.level} {node.num or node.label}"]
        for ch in node.children:
            fill_breadcrumbs(ch, node.breadcrumbs)

    for root in root_nodes:
        fill_breadcrumbs(root, [])

    stats = {"node_count": len(nodes), "leaf_count": sum(1 for n in nodes if not n.children)}
    return ParseResult(doc_type=doc_type, nodes=nodes, root_nodes=root_nodes, stats=stats)
