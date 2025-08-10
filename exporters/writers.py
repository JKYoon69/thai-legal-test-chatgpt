# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict
import io
import json
import zipfile
from datetime import datetime
import re

from parser_core.schema import ParseResult, Chunk
from parser_core.rules_th import normalize_text

def to_jsonl(chunks: List[Chunk]) -> str:
    lines = []
    for c in chunks:
        obj = {
            "text": c.text,
            "span": [c.span_start, c.span_end],
            "node_ids": c.node_ids,
            "breadcrumbs": c.breadcrumbs,
            "meta": c.meta,
        }
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines)

def _span_union_len(chunks: List[Chunk]) -> int:
    ivs = sorted([[c.span_start, c.span_end] for c in chunks], key=lambda x: x[0])
    merged = []
    for s, e in ivs:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return sum(e - s for s, e in merged)

def _count_articles_in_source(src_text: str) -> int:
    norm = normalize_text(src_text)
    NBSP = "\u00A0"
    RE_NUM = r"(?P<num>\d{1,4}(?:/\d{1,3})?)"
    TAIL_NUM = r"(?:\s*\.?)"
    _sp = r"[ \t" + NBSP + r"]+"
    pat_article = re.compile(rf"(?m)^(?P<label>มาตรา){_sp}{RE_NUM}{TAIL_NUM}\b")
    return len(list(pat_article.finditer(norm)))

def _type_counts(chunks: List[Chunk]) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for c in chunks:
        t = c.meta.get("type", "article")
        cnt[t] = cnt.get(t, 0) + 1
    return cnt

def make_debug_report(parse_result: ParseResult, chunks: List[Chunk], source_file: str, law_name: str) -> str:
    span_union = _span_union_len(chunks)
    src_len = len(parse_result.full_text)
    coverage = (span_union / src_len) if src_len else 0.0

    # integrity checks
    overlaps = 0
    mismatches = 0
    last_end = -1
    for c in sorted(chunks, key=lambda x: x.span_start):
        if last_end > -1 and c.span_start < last_end:
            overlaps += 1
        last_end = c.span_end
        if parse_result.full_text[c.span_start:c.span_end] != c.text:
            mismatches += 1

    # article parity
    src_articles = _count_articles_in_source(parse_result.full_text)
    chunk_articles = sum(1 for c in chunks if c.meta.get("type") == "article")

    rpt = {
        "source_file": source_file,
        "law_name": law_name,
        "doc_type": parse_result.doc_type,
        "node_count": len(parse_result.all_nodes),
        "chunk_count": len(chunks),
        "coverage_span_union": round(coverage, 6),
        "integrity": {
            "overlaps": overlaps,
            "text_mismatches": mismatches
        },
        "type_counts": _type_counts(chunks),
        "article_parity": {
            "source_article_count": src_articles,
            "chunk_article_count": chunk_articles
        },
        "sample_chunk": (chunks[0].text[:400] if chunks else ""),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return json.dumps(rpt, ensure_ascii=False, indent=2)

def make_zip_bundle(
    source_text: str,
    parse_result: ParseResult,
    chunks: List[Chunk],
    source_file: str,
    law_name: str,
) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SOURCE.txt", source_text)
        zf.writestr("CHUNKS.jsonl", to_jsonl(chunks))
        zf.writestr("REPORT.json", make_debug_report(parse_result, chunks, source_file, law_name))
    buf.seek(0)
    return buf
