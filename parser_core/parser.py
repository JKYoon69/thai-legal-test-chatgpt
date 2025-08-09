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
    """Find level tokens only at line starts (anchors)."""
    levels = R.LEVELS.get(doc_type, R.LEVELS["unknown"])
    markers = []
    for lv in levels:
        pat = R.RE_HEADER.get(lv)
        if not pat:
            continue
        for m in pat.finditer(text):
            # skip lines starting with brackets
            line_start = text.rfind("\n", 0, m.start()) + 1
            if line_start >= 1 and line_start < m.start():
                if text[line_start] in "([（":
                    continue
            start, end = m.span()
            num = m.group("num")
            markers.append((start, end, lv, f"{lv} {num}"))

    # appendix/tails
    for m in R.RE_APPENDIX.finditer(text):
        start, end = m.span()
        markers.append((start, end, "appendix", text[start:end].strip()))

    markers.sort(key=lambda x: x[0])

    # dedupe very-close markers (defensive)
    deduped = []
    last_pos = -10**9
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

    nodes: List[Node] = []
    root_nodes: List[Node] = []

    # prologue (raw text before first marker)
    if markers and markers[0][0] > 0:
        pro_start = 0
        pro_end = markers[0][0]
        prologue = Node(
            node_id=f"prologue-{pro_start}",
            level="prologue",
            label="prologue",
            num=None,
            span=Span(start=pro_start, end=pro_end),
            breadcrumbs=["prologue"],
            children=[],
        )
        root_nodes.append(prologue)
        nodes.append(prologue)

    if not markers:
        root = Node(
            node_id="document-0",
            level="document",
            label="document",
            num=None,
            span=Span(start=0, end=len(norm)),
            breadcrumbs=["document"],
            children=[],
        )
        return ParseResult(doc_type=doc_type, nodes=[root], root_nodes=[root],
                           stats={"node_count": 1, "leaf_count": 1})

    # compute boundaries
    boundaries = []
    for i, (s, e, lv, label) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(norm)
        boundaries.append((s, e, next_start, lv, label))

    level_order = {lv: i for i, lv in enumerate(R.LEVELS.get(doc_type, R.LEVELS["unknown"]))}
    stack: List[Node] = []

    def _push(node: Node):
        if not stack:
            root_nodes.append(node)
        else:
            stack[-1].children.append(node)
        stack.append(node)

    for s, e, nxt, lv, label in boundaries:
        # rank: lower = higher level
        if lv == "appendix":
            stack.clear()
            lvl_rank = 10**6
        else:
            lvl_rank = level_order.get(lv, 10**5)

        # siblings: pop while top_rank >= current_rank
        while stack:
            top = stack[-1].level
            top_rank = 10**6 if top == "appendix" else level_order.get(top, 10**5)
            if top_rank >= lvl_rank:
                stack.pop()
            else:
                break

        num_match = re.search(r"(\d{1,4}(?:/\d{1,3})?)", label)
        num = num_match.group(1) if num_match else None

        node = Node(
            node_id=f"{lv}-{num or 'x'}-{s}",   # globally unique (includes start offset)
            level=lv,
            label=label,
            num=num,
            span=Span(start=s, end=nxt),
            breadcrumbs=[],
            children=[],
        )
        _push(node)
        nodes.append(node)

    # ---------- Heuristic: group leading articles under a virtual 'front_matter' ----------
    # if before the first higher-level header we have many articles, wrap them.
    higher_levels = {"ภาค", "ลักษณะ", "หมวด", "ส่วน", "บท"}
    first_higher_idx = next((i for i, n in enumerate(root_nodes) if n.level in higher_levels), None)
    if first_higher_idx is not None:
        # collect leading nodes before first higher header that are article/section
        leading_idxs = [i for i, n in enumerate(root_nodes[:first_higher_idx]) if n.level in ("มาตรา", "ข้อ")]
        if leading_idxs:
            count = len(leading_idxs)
            first_article = root_nodes[leading_idxs[0]]
            first_higher = root_nodes[first_higher_idx]
            span_len = first_higher.span.start - first_article.span.start
            if count >= 5 or span_len >= 2000:
                # build virtual parent
                parent = Node(
                    node_id=f"front_matter-{first_article.span.start}",
                    level="front_matter",
                    label="front_matter",
                    num=None,
                    span=Span(start=first_article.span.start, end=first_higher.span.start),
                    breadcrumbs=[],
                    children=[],
                )
                # move targeted children under parent
                moving = [root_nodes[i] for i in leading_idxs]
                for _ in range(len(leading_idxs)):
                    # remove from root in order from the end to keep indices valid
                    root_nodes.pop(leading_idxs[-1])
                    leading_idxs.pop()
                # insert parent before first_higher_idx (which has shifted by number removed)
                insert_pos = next((i for i, n in enumerate(root_nodes) if n is first_higher), len(root_nodes))
                root_nodes.insert(insert_pos, parent)
                parent.children.extend(moving)
                nodes.append(parent)

    # breadcrumbs
    def fill_breadcrumbs(node: Node, trail: List[str]):
        node.breadcrumbs = trail + [f"{node.level} {node.num or node.label}"]
        for ch in node.children:
            fill_breadcrumbs(ch, node.breadcrumbs)

    for root in root_nodes:
        fill_breadcrumbs(root, [])

    stats = {"node_count": len(nodes), "leaf_count": sum(1 for n in nodes if not n.children)}
    return ParseResult(doc_type=doc_type, nodes=nodes, root_nodes=root_nodes, stats=stats)
