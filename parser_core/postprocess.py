# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple

from .schema import ParseResult, Node, Chunk
from .rules_th import normalize_text

# 단계 1: 조문 단위 확보에 집중 — 분할/병합 비활성화(안전 상한만 넉넉히)
MAX_NODE_CHARS = 10000  # 현 샘플의 최대 조문 3,382자 → 분할 발생하지 않도록 충분히 크게

def validate_tree(result: ParseResult) -> List[str]:
    issues: List[str] = []

    # 1) 형제 간 스팬이 겹치지 않는지 점검
    def _check(n: Node):
        for i, c in enumerate(n.children):
            if i + 1 < len(n.children):
                nxt = n.children[i + 1]
                if c.span_end > nxt.span_start:
                    issues.append(f"Span overlap at {c.label} {c.num} -> {nxt.label} {nxt.num}")
            _check(c)
    _check(result.root)

    # 2) 빈 노드 점검(front_matter 제외)
    for n in result.all_nodes:
        if n is result.root or n.label == "front_matter":
            continue
        if len(n.text.strip()) < 2:
            issues.append(f"Empty node: {n.label} {n.num}")

    return issues

def _is_leaf(n: Node) -> bool:
    # 조문/ข้อ를 리프 취급 (하위 노드가 있더라도 현재 단계에선 조문 단위만 청크로 삼음)
    return n.label in ("มาตรา", "ข้อ")

def _gather_leaves(root: Node) -> List[Node]:
    leaves: List[Node] = []
    stack = list(root.children)
    while stack:
        n = stack.pop(0)
        if _is_leaf(n):
            leaves.append(n)
        # 계속 내려가되, 조문이 아닌 상위 노드는 탐색만
        for c in n.children:
            stack.append(c)
    # 문서 순서 보장
    leaves.sort(key=lambda x: x.span_start)
    return leaves

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
    타이틀 라인 + 다음 보조 라인(있으면) 결합
    """
    norm = normalize_text(text)
    lines = [ln.strip() for ln in norm.splitlines()[:60] if ln.strip()]
    title = None
    for i, ln in enumerate(lines):
        if ("พระราชบัญญัติ" in ln) or ("ประมวลกฎหมาย" in ln):
            title = ln
            if i + 1 < len(lines) and len(lines[i + 1]) <= 60 and (
                "ภาค" in lines[i+1] or "ลักษณะ" in lines[i+1] or "พ.ศ." in lines[i+1] or "ฉบับ" in lines[i+1]
            ):
                title = f"{title} / {lines[i+1]}"
            break
    return title

def _split_long_article(text: str, start: int, limit: int) -> List[Tuple[int, int]]:
    """
    조문이 limit을 넘는 드문 경우를 대비한 안전 분할(문단 경계 우선).
    기본적으로 현 문서에서는 분할이 발생하지 않도록 limit을 크게 둠.
    """
    if len(text) <= limit:
        return [(start, start + len(text))]

    parts: List[Tuple[int, int]] = []
    s = 0
    while s < len(text):
        e = min(s + limit, len(text))
        # 문단 경계(\n\n)로 당겨서 자르기
        cut = text.rfind("\n\n", s, e)
        if cut != -1 and cut > s + 200:  # 최소 200자 확보 후 자르기
            e = cut + 2
        parts.append((start + s, start + e))
        s = e
    return parts

def make_chunks(
    result: ParseResult,
    mode: str = "article_only",
    source_file: Optional[str] = None,
    law_name: Optional[str] = None,
) -> List[Chunk]:
    """
    단계 1: '조문 단위' 청크를 1:1로 생성 (필요 시 같은 조문 안에서만 part 분할)
    - 절대 다른 조문끼리 병합하지 않음
    - 커버리지 100%에 최대한 근접(≈99%)하도록 설계
    """
    leaves = _gather_leaves(result.root)
    chunks: List[Chunk] = []

    for leaf in leaves:
        section_label = f"{leaf.label} {leaf.num}".strip()
        meta_base = {
            "mode": "article_only",
            "doc_type": result.doc_type,
            "law_name": law_name or "",
            "section_label": section_label,
            "source_file": source_file or "",
        }

        # 같은 조문 내에서만 필요시 part 분할
        spans = _split_long_article(
            text=result.full_text[leaf.span_start:leaf.span_end],
            start=leaf.span_start,
            limit=MAX_NODE_CHARS,
        )

        for k, (s, e) in enumerate(spans, 1):
            label = section_label + (f" (part {k})" if len(spans) > 1 else "")
            chunks.append(
                Chunk(
                    text=result.full_text[s:e],
                    span_start=s,
                    span_end=e,
                    node_ids=[leaf.node_id],
                    breadcrumbs=_breadcrumbs(leaf, result),
                    meta={**meta_base, "section_label": label},
                )
            )

    # 문서 순서 보장
    chunks.sort(key=lambda c: c.span_start)
    return chunks
