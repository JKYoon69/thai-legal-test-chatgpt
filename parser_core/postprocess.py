from __future__ import annotations
from typing import List
from .schema import ParseResult, Issue, Chunk, Span

def validate_tree(result: ParseResult) -> List[Issue]:
    issues: List[Issue] = []
    for n in result.nodes:
        if n.span.start >= n.span.end:
            issues.append(Issue(level="error", message=f"{n.node_id}: 빈 span"))
    # 번호 중복 간단 체크(같은 레벨/번호가 연속 등장 시 경고)
    seen = set()
    for n in result.nodes:
        if n.num:
            key = (n.level, n.num)
            if key in seen:
                issues.append(Issue(level="warn", message=f"{n.level} 번호 중복: {n.num}"))
            seen.add(key)
    return issues

def make_chunks(raw_text: str, result: ParseResult, mode: str = "article±1") -> List[Chunk]:
    """
    article_only: 'มาตรา/ข้อ' 단위
    article±1  : 이웃 조문과 얕게 병합
    """
    leaves = [n for n in result.nodes if not n.children]
    chunks: List[Chunk] = []

    def is_article(node) -> bool:
        return node.level in ("มาตรา", "ข้อ")

    art_idxs = [i for i, n in enumerate(leaves) if is_article(n)]
    for idx in art_idxs:
        if mode == "article_only":
            chosen = [leaves[idx]]
        else:
            chosen = [x for x in leaves[max(0, idx - 1): idx + 2] if is_article(x)]

        start = min(c.span.start for c in chosen)
        end = max(c.span.end for c in chosen)
        text = raw_text[start:end]

        chunk = Chunk(
            chunk_id=f"chunk-{leaves[idx].node_id}",
            node_ids=[c.node_id for c in chosen],
            text=text.strip(),
            breadcrumbs=leaves[idx].breadcrumbs,
            span=Span(start=start, end=end),
            meta={"mode": mode, "doc_type": result.doc_type},
        )
        chunks.append(chunk)

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
