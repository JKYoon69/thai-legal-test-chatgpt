import json
import zipfile
from typing import Iterable
from parser_core.schema import ParseResult, Chunk

def to_jsonl(objs: Iterable, path):
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o.model_dump(), ensure_ascii=False) + "\n")

def make_zip_bundle(zip_path, files: dict[str, bytes]):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)

def make_debug_report(raw_text: str, result: ParseResult, chunks: list[Chunk]) -> dict:
    """
    Compact debug info to understand mis-splits at a glance.
    """
    def node_to_row(n):
        return {
            "id": n.node_id,
            "level": n.level,
            "label": n.label,
            "num": n.num,
            "span": [n.span.start, n.span.end],
            "len": n.span.end - n.span.start,
            "children_count": len(n.children),
        }

    # first/last 5 nodes for quick view
    nodes_sorted = result.nodes
    head_nodes = [node_to_row(n) for n in nodes_sorted[:5]]
    tail_nodes = [node_to_row(n) for n in nodes_sorted[-5:]]

    # chunk sizes distribution (first few)
    chunk_sizes = [{"id": c.chunk_id, "len": c.span.end - c.span.start, "nodes": c.node_ids[:5]} for c in chunks[:10]]

    return {
        "doc_type": result.doc_type,
        "stats": result.stats,
        "text_length": len(raw_text),
        "nodes_count": len(result.nodes),
        "first_nodes": head_nodes,
        "last_nodes": tail_nodes,
        "chunks_count": len(chunks),
        "sample_chunk_sizes": chunk_sizes,
    }
