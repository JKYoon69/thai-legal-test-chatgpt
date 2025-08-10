# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Any
import io
import json
import zipfile
from datetime import datetime
import re
from collections import Counter

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

def make_zip_bundle(source_text: str, parse_result: ParseResult, chunks: List[Chunk],
                    source_file: str, law_name: str, run_config: Dict[str, Any] | None = None,
                    debug: Dict[str, Any] | None = None) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SOURCE.txt", source_text)
        zf.writestr("CHUNKS.jsonl", to_jsonl(chunks))
        zf.writestr("REPORT.json", make_debug_report(parse_result, chunks, source_file, law_name, run_config, debug))
    buf.seek(0)
    return buf

# ── helpers ── #

def _span_union(chunks: List[Chunk]) -> Tuple[int, List[Tuple[int,int]]]:
    ivs = sorted([[c.span_start, c.span_end] for c in chunks], key=lambda x: x[0])
    merged: List[List[int]] = []
    for s, e in ivs:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return sum(e - s for s, e in merged), [(s,e) for s,e in merged]

def _count_articles_in_source(src_text: str) -> int:
    norm = normalize_text(src_text)
    NBSP = "\u00A0"
    RE_NUM = r"(?P<num>\d{1,4}(?:/\d{1,3})?)"
    TAIL_NUM = r"(?:\s*\.?)"
    _sp = r"[ \t" + NBSP + r"]+"
    pat_article = re.compile(rf"(?m)^(?P<label>มาตรา){_sp}{RE_NUM}{TAIL_NUM}\b")
    return len(list(pat_article.finditer(norm)))

def _overlap_diagnostics(chunks: List[Chunk]) -> Dict[str, Any]:
    chs = sorted(chunks, key=lambda c: (c.span_start, c.span_end))
    overlaps=0
    by_pair: Counter = Counter()
    last_s=last_e=None; last_t="?"
    for c in chs:
        s,e = c.span_start, c.span_end; t = c.meta.get("type","article")
        if last_s is not None and s < last_e:
            overlaps += 1; by_pair[(last_t, t)] += 1
        last_s, last_e, last_t = s,e,t
    return {"overlaps": overlaps,
            "overlap_pairs_by_type": {f"{a}->{b}": cnt for (a,b),cnt in by_pair.items()}}

def _tree_stats(parse_result: ParseResult) -> Dict[str, Any]:
    by_label: Counter = Counter(n.label or "?" for n in parse_result.all_nodes)
    depth_max = max((n.level for n in parse_result.all_nodes), default=0)
    empty_nodes = [f"{n.label} {n.num}".strip() for n in parse_result.all_nodes if (n.text is not None and len(n.text.strip()) == 0)]
    span_issues = []
    for p in parse_result.all_nodes:
        for ch in p.children:
            if ch.span_start < p.span_start or ch.span_end > p.span_end:
                span_issues.append(f"child outside parent: {ch.label} {ch.num} in {p.label} {p.num}")
    return {"node_count": len(parse_result.all_nodes),
            "by_label": dict(by_label),
            "max_depth": depth_max,
            "empty_nodes": empty_nodes[:20],
            "span_issues": span_issues[:20]}

def _duplicates(chunks: List[Chunk]) -> Dict[str, Any]:
    import hashlib
    hash_to_idxs: Dict[str, List[int]] = {}
    region_key: Dict[Tuple[Tuple[str,...], Tuple[int,int]], List[int]] = {}
    for i,c in enumerate(chunks):
        h = hashlib.sha1(c.text.encode("utf-8")).hexdigest()
        hash_to_idxs.setdefault(h, []).append(i)
        k = (tuple(c.breadcrumbs or []), (c.span_start, c.span_end))
        region_key.setdefault(k, []).append(i)
    exact_groups = sorted([(h, idxs) for h,idxs in hash_to_idxs.items() if len(idxs)>1], key=lambda x: len(x[1]), reverse=True)
    region_dups = sorted([(k, idxs) for k,idxs in region_key.items() if len(idxs)>1], key=lambda x: len(x[1]), reverse=True)
    return {"exact_dup_groups": len(exact_groups),
            "region_dup_groups": len(region_dups),
            "exact_dup_top": [{"count": len(idxs)} for h,idxs in exact_groups[:5]]}

def _largest_gaps(full_text: str, chunks: List[Chunk], topn: int = 5) -> List[Dict[str, Any]]:
    total = len(full_text)
    ivs = sorted([[c.span_start, c.span_end] for c in chunks], key=lambda x: x[0])
    merged = []
    for s,e in ivs:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    gaps = []; prev = 0
    for s,e in merged:
        if s>prev: gaps.append((prev, s))
        prev=e
    if prev < total: gaps.append((prev, total))
    def snip(s,e): return full_text[s:min(s+200, len(full_text))].replace("\n","⏎")
    gaps_sorted = sorted(gaps, key=lambda g: (g[1]-g[0]), reverse=True)[:topn]
    return [{"span": [s,e], "len": e-s, "preview": snip(s,e)} for s,e in gaps_sorted]

# ── 조문 크기 통계 ── #

def _percentiles(vals: List[int]) -> Dict[str, float]:
    if not vals: return {"p25":0,"p50":0,"p75":0,"min":0,"max":0,"mean":0.0}
    s = sorted(vals)
    def q(p: float) -> float:
        k = (len(s)-1)*p; a = int(k); b = a+1
        if b >= len(s): return float(s[a])
        return float(s[a] + (s[b]-s[a])*(k-a))
    return {"min": float(s[0]), "p25": q(0.25), "p50": q(0.50),
            "p75": q(0.75), "max": float(s[-1]), "mean": float(sum(s)/len(s))}

def _histogram(vals: List[int]) -> Dict[str, int]:
    buckets = [(0,200),(200,500),(500,1000),(1000,2000),(2000,5000),(5000,10**9)]
    names = ["0-200","200-500","500-1000","1000-2000","2000-5000","5000+"]
    counts = {n:0 for n in names}
    for v in vals:
        for (lo,hi),name in zip(buckets,names):
            if lo <= v < hi: counts[name] += 1; break
    return counts

def _article_size_stats(chunks: List[Chunk]) -> Dict[str, Any]:
    arts = [c for c in chunks if c.meta.get("type","article")=="article"]
    lens = [len(c.text) for c in arts]
    sizes = _percentiles(lens)
    hist = _histogram(lens)
    by_series: Counter = Counter(c.meta.get("series","1") for c in arts)
    arts_sorted = sorted(arts, key=lambda c: len(c.text))
    shortest = [{"section_label": c.meta.get("section_label",""), "len": len(c.text)} for c in arts_sorted[:5]]
    longest  = [{"section_label": c.meta.get("section_label",""), "len": len(c.text)} for c in arts_sorted[-5:]][::-1]
    return {"count": len(arts), "length_stats_chars": sizes, "length_histogram": hist,
            "series_counts": dict(by_series), "top_shortest": shortest, "top_longest": longest}

# ── REPORT ── #

def make_debug_report(parse_result: ParseResult, chunks: List[Chunk], source_file: str,
                      law_name: str, run_config: Dict[str, Any] | None = None,
                      debug: Dict[str, Any] | None = None) -> str:
    full = parse_result.full_text
    union_len, merged = _span_union(chunks)
    src_len = len(full)
    coverage = (union_len / src_len) if src_len else 0.0

    integ = _overlap_diagnostics(chunks)
    mismatches = 0
    for c in chunks:
        if full[c.span_start:c.span_end] != c.text: mismatches += 1
    integ["text_mismatches"] = mismatches

    type_counts = Counter(c.meta.get("type","article") for c in chunks)
    type_sizes = Counter()
    for c in chunks:
        type_sizes[c.meta.get("type","article")] += len(c.text)

    mk_diag = (debug or {}).get("make_chunks_diag", {}) if debug else {}
    strict_post_fill = mk_diag.get("strict_post_fill", {"enabled": False, "filled_gaps": 0, "total_chars": 0, "spans": []})
    short_headnotes_removed = mk_diag.get("short_headnotes_removed", 0)

    report = {
        "source_file": source_file,
        "law_name": law_name,
        "doc_type": parse_result.doc_type,
        "run_config": run_config or {},
        "tree": _tree_stats(parse_result),
        "chunks": {
            "count": len(chunks),
            "type_counts": dict(type_counts),
            "type_sizes": dict(type_sizes),
        },
        "coverage": {
            "source_len_chars": src_len,
            "union_len_chars": union_len,
            "coverage_span_union": round(coverage, 6),
            "largest_gaps_top5": _largest_gaps(full, chunks, topn=5),
        },
        "integrity": integ,
        "duplicates": _duplicates(chunks),
        "article_parity": {
            "source_article_count": _count_articles_in_source(full),
            "chunk_article_count": sum(1 for c in chunks if c.meta.get("type","article")=="article")
        },
        "article_size_stats": _article_size_stats(chunks),
        "strict_post_fill": strict_post_fill,
        "short_headnotes_removed": short_headnotes_removed,
        "sample_chunk_head": (chunks[0].text[:400] if chunks else ""),
        "debug": debug or {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return json.dumps(report, ensure_ascii=False, indent=2)
