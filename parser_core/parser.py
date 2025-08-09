from __future__ import annotations
from typing import List, Tuple
import regex as re
from .schema import Node, Span, ParseResult
from . import rules_th as R


def detect_doc_type(text: str) -> str:
    t = text[:5000]  # 헤더 주변만으로 판정 충분
    t = t.translate(R.THAI2ARABIC)
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
    라벨 토큰들을 전부 찾아 (start, end, level, label_with_num) 리스트로 반환
    """
    levels = R.LEVELS.get(doc_type, R.LEVELS["unknown"])
    markers = []
    for lv in levels:
        pat = R.RE_HEADER.get(lv)
        if not pat:
            continue
        for m in pat.finditer(text):
            start, end = m.span()
            num = m.group("num")
            label = f"{lv} {num}"
            markers.append((start, end, lv, label))
    # 부록
    for m in R.RE_APPENDIX.finditer(text):
        start, end = m.span()
        markers.append((start, end, "appendix", text[start:end].strip()))
    markers.sort(key=lambda x: x[0])
    return markers


def parse_document(text: str, forced_type: str | None = None) -> ParseResult:
    norm = R.normalize_text(text)
    doc_type = forced_type or detect_doc_type(norm)

    markers = _find_all_markers(norm, doc_type)
    # 시작/끝 경계 보정: 마지막 마커 이후 ~ 문서 끝
    boundaries = []
    for i, (s, e, lv, label) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(norm)
        boundaries.append((s, e, next_start, lv, label))

    nodes: List[Node] = []
    # 스택으로 계층 구성
    level_order = {lv: i for i, lv in enumerate(R.LEVELS.get(doc_type, R.LEVELS["unknown"]))}
    root_nodes: List[Node] = []
    stack: List[Node] = []

    def _push(node: Node):
        if not stack:
            root_nodes.append(node)
        else:
            stack[-1].children.append(node)
        stack.append(node)

    for s, e, nxt, lv, label in boundaries:
        # appendix는 최하위로 취급(문서 끝까지)
        lvl_rank = 999 if lv == "appendix" else level_order.get(lv, 998)

        # 현재 스택 상단보다 상위 레벨이면 pop
        while stack:
            top_rank = 999 if stack[-1].level == "appendix" else level_order.get(stack[-1].level, 998)
            if top_rank <= lvl_rank:
                break
            stack.pop()

        # 숫자 추출
        m = re.search(r"(\d{1,4}(?:/\d{1,3})?)", label)
        num = m.group(1) if m else None

        node = Node(
            node_id=f"{lv}-{num or len(nodes)+1}",
            level=lv,
            label=label,
            num=num,
            span=Span(start=s, end=nxt),
            text=norm[s:nxt].strip(),
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

    stats = {
        "node_count": len(nodes),
        "leaf_count": sum(1 for n in nodes if not n.children),
    }
    return ParseResult(doc_type=doc_type, nodes=nodes, root_nodes=root_nodes, stats=stats)
