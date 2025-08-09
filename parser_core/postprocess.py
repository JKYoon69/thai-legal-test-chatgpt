from __future__ import annotations
from typing import List, Dict, Tuple
from .schema import ParseResult, Issue, Chunk, Span, Node

def _check_duplicates_scoped_by_parent(parent: Node, issues: List[Issue]):
    """
    Duplicate number check per parent scope.
    E.g., 'มาตรา 1..' under the Act and 'มาตรา 1..' under the Code are NOT duplicates.
    """
    # group siblings by level
    by_level: Dict[str, Dict[str, int]] = {}
    for ch in parent.children:
        if ch.num:
            by_level.setdefault(ch.level, {})
            key = ch.num
            by_level[ch.level][key] = by_level[ch.level].get(key, 0) + 1

    for level, counter in by_level.items():
        for num, cnt in counter.items():
            if cnt > 1:
                issues.append(Issue(level="warn", message=f"Duplicate {level} number within same parent: {num} (x{cnt})"))

    # recurse
    for ch in parent.children:
        _check_duplicates_scoped_by_parent(ch, issues)

def validate_tree(result: ParseResult) -> List[Issue]:
    issues: List[Issue] = []
    # basic span sanity
    for n in result.nodes:
        if n.span.start >= n.span.end:
            issues.append(Issue(level="error", message=f"{n.node_id}: empty span"))

    # scoped duplicate check (parent-wise)
    for root in result.root_nodes:
        _check_duplicates_scoped_by_parent(root, issues)

    return issues

def make_chunks(raw_text: str, result: ParseResult, mode: str = "article±1") -> List[Chunk]:
    """
    article_only: 'มาตรา/ข้อ' unit
    article±1  : shallow-merge with neighbors
    fallback   : leaf-based chunking if no article found
    """
    leaves = [n for n in result.nodes if not n.children]
    chunks: List[Chunk] = []

    def is_article(node) -> bool:
        return node.level in ("มาตรา", "ข้อ")

    art_idxs = [i for i, n in enumerate(leaves) if is_article(n)]
    for idx in art_idxs:
        chosen = [leaves[idx]] if mode == "article_only" else \
                 [x for x in leaves[max(0, idx - 1): idx + 2] if is_article(x)]

        start = min(c.span.start for c in chosen)
        end = max(c.span.end for c in chosen)
        text = raw_text[start:end]

        chunks.append(
            Chunk(
                chunk_id=f"chunk-{leaves[idx].node_id}",
                node_ids=[c.node_id for c in chosen],
                text=text.strip(),
                breadcrumbs=leaves[idx].breadcrumbs,
                span=Span(start=start, end=end),
                meta={"mode": mode, "doc_type": result.doc_type},
            )
        )

    if not chunks and leaves:
        for leaf in leaves:
            chunks.append(
                Chunk(
                    chunk_id=f"chunk-{leaf.node_id}",
                    node_ids=[leaf.node_id],
                    text=raw_text[leaf.span.start:leaf.span.end],
                    breadcrumbs=leaf.breadcrumbs,
                    span=leaf.span,
                    meta={"mode": "leaf", "doc_type": result.doc_type},
                )
            )
    return chunks
