# -*- coding: utf-8 -*-
"""
Thai Legal Preprocessor — RAG-ready (lossless + debug)

Safe changes / what's new:
- English UI.
- One-click "Download All" (ZIP with rich.jsonl, compat.jsonl, REPORT.json).
- "New File" button resets the whole app state and uploader.
- Structure-free fallback chunking (paragraph+length) for structureless docs only.
- Enumeration boundary snap with coverage repair + label/UID reconciliation when anchors disagree.
- Zero-length cleanup; overlap visualization; LLM usage/cost in REPORT.
- Mode flag ("structured" | "fallback") recorded in report.
"""

import os, sys, re, time, json, math, io, zipfile, hashlib, random
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st

# --- Path fix ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Parser / chunker core
from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import (
    validate_tree, make_chunks, guess_law_name, repair_tree,
    merge_small_trailing_parts,
)
from parser_core.schema import ParseResult, Chunk

# Writers (compatibility import path)
try:
    from writers import to_jsonl_rich_meta, to_jsonl_compat_flat, make_debug_report
except ModuleNotFoundError:
    from exporters.writers import to_jsonl_rich_meta, to_jsonl_compat_flat, make_debug_report

# LLM (batch + cache)
from llm_clients import call_openai_batch
from extractors import build_summary_record

APP_TITLE = "Thai Legal Preprocessor — RAG-ready (lossless + debug)"

# ----------------------- Utilities -----------------------
def _esc(s:str)->str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _coverage(chunks: List, total_len: int) -> float:
    ivs = sorted([[int(c.span_start), int(c.span_end)] for c in chunks], key=lambda x:x[0])
    merged=[]
    for s,e in ivs:
        if not merged or s>merged[-1][1]: merged.append([s,e])
        else: merged[-1][1]=max(merged[-1][1], e)
    covered=sum(e-s for s,e in merged)
    return (covered/total_len) if total_len else 0.0

def _basename_no_ext(name: str) -> str:
    base = name or "document"
    if "." in base: base = base.rsplit(".",1)[0]
    return base

def _normalize_law_name(name: Optional[str], fallback_filename: str) -> str:
    """Tighten weird spacing and backfill from filename if it looks too short."""
    if not name: name = ""
    # collapse weird Thai spacing artifacts
    name = re.sub(r"\s+", " ", name).strip()
    # if name too short, fall back to filename (minus extension)
    if len(name) < 6:
        name = _basename_no_ext(fallback_filename)
    # ensure year / ฉบับที่ kept contiguous
    name = re.sub(r"(พ\.ศ\.)\s+(\d{4})", r"\1 \2", name)
    name = re.sub(r"(ฉบับที่)\s+(\d+)", r"\1 \2", name)
    return name

# ----------------------- Fallback chunking -----------------------
class SimpleChunk:
    """Duck-typed Chunk enough for writers/exporters."""
    def __init__(self, text: str, span_start: int, span_end: int, meta: Dict[str,Any], breadcrumbs: Optional[List[str]]=None):
        self.text = text
        self.span_start = span_start
        self.span_end = span_end
        self.meta = meta or {}
        self.breadcrumbs = breadcrumbs or []

def _paragraph_breaks(text: str) -> List[int]:
    """
    Paragraph break candidates: empty lines (double newlines). If none, allow single newline.
    Returns absolute offsets (core end candidates).
    """
    N = len(text)
    brks = set()
    for m in re.finditer(r'(?:\r?\n[ \t\f\v]*\r?\n)+', text):
        brks.add(m.end())
    if not brks:
        for m in re.finditer(r'\r?\n', text):
            brks.add(m.end())
    brks.add(N)
    return sorted(brks)

def _second_cut_candidates(seg: str) -> List[int]:
    """
    Optional inner split hints for very long paragraphs:
    - list items like '1) ', 'ข้อ 3', bullets
    - punctuation hints like ';' or ':' followed by newline/space
    Returns list of local offsets (within seg) where a cut could happen.
    """
    cand = set()
    # list numbers or Thai article indicators
    for m in re.finditer(r'(?m)^(?:\s*(?:\d{1,4}|[\u0E50-\u0E59]{1,4})\)\s+|(?:ข้อ|มาตรา)\s*(?:\d{1,4}|[\u0E50-\u0E59]{1,4})\b|[•·\-–—]\s+)', seg):
        cand.add(m.start())
    # punctuation hints
    for m in re.finditer(r'[;:]\s+(?=\S)', seg):
        cand.add(m.end())
    return sorted(cand)

def make_generic_chunks(full_text: str, *, law_name: str, source_file: str,
                        target_chars: int, min_ratio: float, overlap_chars: int,
                        tail_merge_min_chars: int, enable_second_cut: bool=True) -> Tuple[List[SimpleChunk], Dict[str,Any]]:
    """
    Structure-free length-based split:
    - Prefer paragraph boundaries within [pos+min, pos+target]
    - If not available, extend search window slightly; otherwise hard-cut
    - Optional second-cut hints inside very long paragraphs
    - Tail merge if last piece too short
    """
    N = len(full_text)
    breaks = _paragraph_breaks(full_text)
    cores: List[Tuple[int,int]] = []
    pos = 0
    min_chars = max(200, int(target_chars * min_ratio))
    ext = 160  # extra window to find a nice boundary

    while pos < N:
        lo = pos + min_chars
        hi = min(N, pos + target_chars)
        cand = [b for b in breaks if lo <= b <= hi]
        cut = None
        if cand:
            cut = max(cand)
        else:
            # try extended window
            ext_hi = min(N, hi + ext)
            cand2 = [b for b in breaks if hi < b <= ext_hi]
            cut = (min(N, pos + target_chars) if not cand2 else min(max(cand2), N))

        if cut is None or cut <= pos:
            cut = min(N, pos + target_chars)
            if cut <= pos:
                break

        # Optional inner hint to avoid overly long paragraph cores
        if enable_second_cut and (cut - pos) > int(1.25 * target_chars):
            local = full_text[pos:cut]
            hints = _second_cut_candidates(local)
            # choose the furthest hint after min_chars but before target_chars if possible
            good = [pos + h for h in hints if (pos + h) >= lo and (pos + h) <= (pos + target_chars)]
            if good:
                cut = max(good)

        cores.append((pos, cut))
        pos = cut

    # tail merge
    if len(cores) >= 2 and (cores[-1][1] - cores[-1][0]) < tail_merge_min_chars:
        cores[-2] = (cores[-2][0], cores[-1][1])
        cores.pop()

    chunks: List[SimpleChunk] = []
    for i,(cs,ce) in enumerate(cores, start=1):
        s = max(0, cs - (overlap_chars if i>1 else 0))
        e = min(N, ce + (overlap_chars if i < len(cores) else 0))
        meta = {
            "type": "article",  # downstream compatibility
            "law_name": law_name or "",
            "source_file": source_file or "",
            "section_label": f"part {i}",
            "section_uid": f"part {i}|{cs}",
            "core_span": [cs, ce],
            "overlap_left": cs - s,
            "overlap_right": e - ce,
            "doc_type": "unknown",
        }
        text = full_text[s:e]
        chunks.append(SimpleChunk(text=text, span_start=s, span_end=e, meta=meta, breadcrumbs=[]))

    diag = {
        "made": True, "strategy":"paragraph+length",
        "cores": len(cores), "chunks": len(chunks),
        "target_chars": target_chars, "min_chars": min_chars, "overlap_chars": overlap_chars,
        "tail_merge_min_chars": tail_merge_min_chars, "second_cut": bool(enable_second_cut)
    }
    return chunks, diag

# ----------------------- Enumeration snap + label reconciliation -----------------------
# Thai digits range: \u0E50-\u0E59
ENUM_PATTERNS = [
    ("digit", re.compile(r'^\s*[\(\[\{]?\s*(?:\d{1,4}|[\u0E50-\u0E59]{1,4})\s*[\)\]\}]?\s*')),
    ("lawterm", re.compile(r'^\s*(?:ข้อ|มาตรา)\s*(?:\d{1,4}|[\u0E50-\u0E59]{1,4})\s*')),
    ("bullet", re.compile(r'^\s*(?:•|·|-|—|–)\s+')),
]

THAI_DIGITS = {ord('๐'): '0', ord('๑'): '1', ord('๒'): '2', ord('๓'): '3', ord('๔'): '4',
               ord('๕'): '5', ord('๖'): '6', ord('๗'): '7', ord('๘'): '8', ord('๙'): '9'}

def _line_start(text: str, pos: int) -> int:
    lb = text.rfind("\n", 0, pos)
    return 0 if lb < 0 else lb + 1

def _detect_enum_anchor_at_chunk_start(text: str, s_next: int, lookahead: int = 80) -> Optional[Tuple[int,int,str]]:
    N = len(text)
    ls = _line_start(text, s_next)
    b = min(N, s_next + lookahead)
    seg = text[ls:b]
    for kind, rx in ENUM_PATTERNS:
        m = rx.match(seg)
        if m:
            return (ls + m.start(), ls + m.end(), kind)
    return None

def _refresh_text(c, full_text: str) -> None:
    c.text = full_text[int(c.span_start):int(c.span_end)]

def _parse_anchor_label(anchor_text: str) -> Optional[str]:
    """Return normalized section label like 'มาตรา 12' or 'ข้อ 3' if detectable."""
    t = anchor_text.strip()
    m = re.match(r'^(ข้อ|มาตรา)\s*([\d\u0E50-\u0E59]{1,4})', t)
    if not m:
        return None
    head = m.group(1)
    num = m.group(2).translate(THAI_DIGITS)
    return f"{head} {num}"

def normalize_enumeration_boundaries(chunks, full_text: str, *, reconcile_labels: bool=True) -> Dict[str, Any]:
    """
    Use enumeration/bullet anchors on the next-chunk line start:
    - prev.span_end = enum_start  (trim tail before the anchor)
    - next.span_start = enum_start (close gaps)
    - next.core_span[0] = enum_start (include anchor into core)
    - (optional) reconcile next.meta.section_label with the anchor
    """
    diag = {
        "pairs_seen": 0, "hits": 0, "cut_only": 0, "shift_only": 0, "both": 0,
        "pattern_hits": {"digit":0,"lawterm":0,"bullet":0},
        "moved_chars_hist": [], "adjusted_next_start": 0, "core_moved_to_enum": 0,
        "label_mismatch_fixed": 0, "label_samples": []
    }
    if not chunks: return diag

    units = [c for c in chunks if (c.meta or {}).get("type") in ("article","article_pack")]
    units.sort(key=lambda c:(int(c.span_start), int(c.span_end)))

    for i in range(1, len(units)):
        prev = units[i-1]; cur = units[i]
        diag["pairs_seen"] += 1

        s_next = int(cur.span_start)
        hit = _detect_enum_anchor_at_chunk_start(full_text, s_next)
        if not hit:
            continue
        enum_s, enum_e, kind = hit

        move_prev = abs(int(prev.span_end) - enum_s)
        move_cur_start = abs(s_next - enum_s)
        cs0, ce0 = (cur.meta or {}).get("core_span", [int(cur.span_start), int(cur.span_end)])
        move_cur_core = abs(int(cs0) - enum_s)
        if move_prev > 120 and move_cur_start > 120 and move_cur_core > 120:
            continue

        applied_cut=False; applied_shift=False; applied_next_start=False

        # 1) trim previous tail
        if int(prev.span_end) > enum_s >= int(prev.span_start):
            prev.span_end = enum_s
            pcs, pce = (prev.meta or {}).get("core_span", [int(prev.span_start), int(prev.span_end)])
            if pce > prev.span_end: pce = prev.span_end
            if pcs < prev.span_start: pcs = prev.span_start
            prev.meta["core_span"] = [int(pcs), int(pce)]
            prev.meta["overlap_right"] = max(0, int(prev.span_end) - int(pce))
            _refresh_text(prev, full_text)
            applied_cut=True

        # 2) shift next.start to enum_s
        if enum_s < int(cur.span_start):
            cur.span_start = enum_s
            _refresh_text(cur, full_text)
            diag["adjusted_next_start"] += 1
            applied_next_start=True

        # 3) include anchor into next.core
        cs, ce = (cur.meta or {}).get("core_span", [int(cur.span_start), int(cur.span_end)])
        cs_new = enum_s
        if cs_new < int(cs):
            cur.meta["core_span"] = [int(cs_new), int(ce)]
            cur.meta["overlap_left"] = max(0, int(cs_new) - int(cur.span_start))
            applied_shift=True
            diag["core_moved_to_enum"] += 1

        # 4) optional: reconcile label with anchor
        if reconcile_labels and kind == "lawterm":
            anchor_label = _parse_anchor_label(full_text[enum_s:enum_e] or "")
            if anchor_label:
                old_label = (cur.meta or {}).get("section_label","")
                if old_label.strip() != anchor_label.strip():
                    # keep history
                    prev_labels = (cur.meta or {}).get("prev_labels", [])
                    if old_label and old_label not in prev_labels:
                        prev_labels = prev_labels + [old_label]
                    cur.meta["prev_labels"] = prev_labels
                    # set new label and UID
                    cur.meta["section_label"] = anchor_label
                    # keep part suffix if existed, e.g., "(part 2)"
                    m = re.search(r"\(part\s*\d+\)", old_label)
                    if m:
                        cur.meta["section_label"] += f" {m.group(0)}"
                    uid_suffix = cur.meta.get("core_span",[int(cur.span_start), int(cur.span_end)])[0]
                    cur.meta["section_uid"] = f"{cur.meta['section_label']}|{uid_suffix}"
                    diag["label_mismatch_fixed"] += 1
                    if len(diag["label_samples"]) < 5:
                        diag["label_samples"].append({"old": old_label, "new": cur.meta["section_label"]})

        if applied_cut or applied_shift or applied_next_start:
            diag["hits"] += 1
            diag["pattern_hits"][kind] = diag["pattern_hits"].get(kind,0) + 1
            both = applied_cut and applied_shift
            if both: diag["both"] += 1
            elif applied_cut and not applied_shift: diag["cut_only"] += 1
            elif applied_shift and not applied_cut: diag["shift_only"] += 1
            mv = 0
            if applied_cut: mv += move_prev
            if applied_shift: mv += move_cur_core
            if applied_next_start: mv += move_cur_start
            diag["moved_chars_hist"].append(min(mv, 400))

    return diag

def fix_zero_length_and_gaps(chunks: List, full_text: str) -> Dict[str, Any]:
    """
    Clean zero-length chunks and pull next.start back to cover potential gaps left by snapping.
    """
    diag = {"removed_zero_span":0, "fixed_next_start":0, "removed_labels":[]}
    chunks.sort(key=lambda c:(int(c.span_start), int(c.span_end)))
    i=0
    while i < len(chunks):
        c = chunks[i]
        s,e = int(c.span_start), int(c.span_end)
        if e <= s:
            if i+1 < len(chunks):
                nxt = chunks[i+1]
                if int(nxt.span_start) > s:
                    nxt.span_start = s
                    _refresh_text(nxt, full_text)
                    cs, ce = (nxt.meta or {}).get("core_span", [int(nxt.span_start), int(nxt.span_end)])
                    nxt.meta["overlap_left"] = max(0, int(cs) - int(nxt.span_start))
                    diag["fixed_next_start"] += 1
            diag["removed_zero_span"] += 1
            diag["removed_labels"].append((c.meta or {}).get("section_label",""))
            del chunks[i]
            continue
        i+=1
    return diag

# ----------------------- Visualization -----------------------
def _inject_css():
    st.markdown("""
    <style>
      .block-container { max-width: 1400px !important; padding-top: .6rem; }
      .docwrap { border:1px solid rgba(107,114,128,.25); border-radius:10px; padding:16px; }
      .doc { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
             font-size:13.5px; line-height:1.6; position:relative; }
      .chunk-wrap { position:relative; padding:0 12px; border-radius:6px; margin:0 1px; }
      .chunk-wrap::before, .chunk-wrap::after {
        content:"{"; position:absolute; top:-1px; bottom:-1px; width:10px;
        color:#b8f55a; opacity:.95; font-weight:800;
      }
      .chunk-wrap::before { left:0; }
      .chunk-wrap::after  { right:0; transform:scaleX(-1); }
      .overlap { background: rgba(244,114,182,.22); }
      .overlap-mark { color:#ec4899; font-weight:800; }
      .close-tail { color:#b8f55a; font-weight:800; }
      .dlbar { display:flex; gap:10px; align-items:center; margin:.6rem 0 .6rem; flex-wrap:wrap; }
      .parse-line { margin:.25rem 0 .6rem; display:flex; gap:.5rem; }
      .parse-line button { background:#22c55e !important; color:#fff !important; border:0 !important;
                           padding:.6rem 1rem !important; font-weight:800 !important; font-size:15px !important;
                           border-radius:10px !important; }
      .danger { background:#ef4444 !important; }
      .accent { background:#3b82f6 !important; }
      .badge-warn { display:inline-block; padding:2px 8px; border-radius:8px; background:#fee2e2; color:#b91c1c; font-weight:700; }
      .muted { color:#9ca3af; font-size:12px; }
      .kv { font-size:12.5px; color:#93c5fd; }
    </style>
    """, unsafe_allow_html=True)

def render_with_overlap(text: str, chunks: List, *, include_front=True, include_head=True) -> str:
    N=len(text)
    units=[]
    if include_front:
        units += [c for c in chunks if (c.meta or {}).get("type")=="front_matter"]
    if include_head:
        units += [c for c in chunks if (c.meta or {}).get("type")=="headnote"]
    units += [c for c in chunks if (c.meta or {}).get("type") in ("article","article_pack")]
    units.sort(key=lambda c:(int(c.span_start),int(c.span_end)))

    parts=[]; cur=0; idx=0
    for c in units:
        s,e=int(c.span_start),int(c.span_end)
        if s<0 or e>N or e<=s: continue
        if s>cur: parts.append(_esc(text[cur:s]))

        cs,ce = (c.meta or {}).get("core_span",[s,e])
        if not (isinstance(cs,int) and isinstance(ce,int) and s<=cs<=ce<=e):
            cs,ce=s,e

        parts.append(f'<span class="chunk-wrap" title="{_esc((c.meta or {}).get("section_label","chunk"))}">')
        parts.append(_esc(text[s:cs]))               # left overlap (plain)
        parts.append(_esc(text[cs:ce]))              # core
        if e>ce:                                     # right overlap (pink)
            over_txt = text[ce:e]
            parts.append(f'<span class="overlap">{_esc(over_txt)}</span>')
            if (e - ce) > 4:
                parts.append('<span class="overlap-mark">}</span>')
        parts.append("</span>")

        idx+=1
        brief = (c.meta or {}).get("brief")
        if not brief:
            brief = (text[cs: min(ce, cs+180)] or "").replace("\n"," ").strip()
        tail = f' }} chunk {idx:04d}'
        if brief:
            tail += f' — { _esc(brief) }'
        parts.append(f' <span class="close-tail">{tail}</span><br/>')
        cur=e
    if cur<N: parts.append(_esc(text[cur:N]))
    return '<div class="doc">'+"".join(parts)+"</div>"

# ----------------------- App state -----------------------
def _defaults() -> Dict[str, Any]:
    return {
        "text":None, "source_file":None, "parsed":False,
        "doc_type":None, "law_name":None, "result":None, "chunks":[], "issues":[],
        "strict_lossless":True, "split_long_articles":True,
        "split_threshold_chars":1500, "tail_merge_min_chars":200, "overlap_chars":200,
        "bracket_front_matter":True, "bracket_headnotes":True,
        # fallback-only knobs
        "fallback_enable": True,
        "fallback_target_chars": 1200,
        "fallback_min_ratio": 0.40,   # min_chars = target * this
        "fallback_second_cut": True,
        # LLM options
        "use_llm_summary":True, "skip_short_chars":180, "batch_group_size":8, "parallel_calls":3, "max_calls":0,
        "llm_cache":{}, "llm_errors":[],
        # outputs (strings)
        "report_json_str":"", "chunks_jsonl_rich":"", "chunks_jsonl_flat":"",
        # UI
        "upload_key": str(random.random()),
    }

def _ensure_state():
    for k,v in _defaults().items():
        st.session_state.setdefault(k,v)

def _reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state.update(_defaults())
    st.rerun()

class RunPanel:
    def __init__(self):
        with st.sidebar:
            st.header("Progress")
            self.status = st.status("Idle...", expanded=True)
            self.prog = st.progress(0, text="Idle")
            self.lines = st.container()
        self.t0 = time.time(); self.done=0
    def _update(self, label, inc=None, set_to=None):
        if set_to is not None: self.done=set_to
        elif inc is not None: self.done=min(100, self.done+inc)
        self.prog.progress(int(self.done), text=f"{label} · {int(self.done)}%")
    def step(self, label, state="running"): self.status.update(label=label, state=state)
    def log(self, text): 
        with self.lines: st.markdown(f"- {text}")
    def tick(self, label, inc): self._update(label, inc=inc)
    def finalize(self, ok=True):
        self.status.update(label=f"Done · {time.time()-self.t0:.1f}s", state=("complete" if ok else "error"))
        self._update("Done", set_to=100)

# ----------------------- Main -----------------------
def main():
    _inject_css(); _ensure_state()
    st.title(APP_TITLE)

    # Top action row: New File + Parse
    st.markdown('<div class="parse-line">', unsafe_allow_html=True)
    new_file = st.button("New File", key="btn_new", type="secondary")
    parse_btn = st.button("Parse", key="parse_btn_top")
    st.markdown('</div>', unsafe_allow_html=True)

    if new_file:
        _reset_all()
        return

    # File uploader (resettable via key)
    up = st.file_uploader("Upload a text file (.txt, UTF-8)", type=["txt"], key=st.session_state["upload_key"])

    # Options — Chunking
    st.subheader("Chunking Options")
    r1,r2,r3 = st.columns(3)
    with r1:
        st.session_state["split_threshold_chars"]=st.number_input("Split threshold (chars)",600,6000,st.session_state["split_threshold_chars"],100)
    with r2:
        st.session_state["tail_merge_min_chars"]=st.number_input("Tail merge min length",0,600,st.session_state["tail_merge_min_chars"],10)
    with r3:
        st.session_state["overlap_chars"]=st.number_input("Overlap (context) length",0,800,st.session_state["overlap_chars"],25)

    o1,o2,o3 = st.columns(3)
    with o1:
        st.session_state["strict_lossless"]=st.checkbox("Strict lossless", value=st.session_state["strict_lossless"])
    with o2:
        st.session_state["split_long_articles"]=st.checkbox("Split extra-long articles", value=st.session_state["split_long_articles"])
    with o3:
        st.session_state["bracket_front_matter"]=st.checkbox("Show front matter", value=st.session_state["bracket_front_matter"])
        st.session_state["bracket_headnotes"]=st.checkbox("Show headnotes (หมวด/ส่วน/etc.)", value=st.session_state["bracket_headnotes"])

    # Fallback-only knobs
    st.subheader("Structure-free Fallback (for structureless docs)")
    f1,f2,f3,f4 = st.columns(4)
    with f1:
        st.session_state["fallback_enable"]=st.checkbox("Enable fallback", value=st.session_state["fallback_enable"])
    with f2:
        st.session_state["fallback_target_chars"]=st.number_input("Fallback target (chars)", 800, 2400, st.session_state["fallback_target_chars"], 50)
    with f3:
        st.session_state["fallback_min_ratio"]=st.number_input("Fallback min ratio", 0.20, 0.80, st.session_state["fallback_min_ratio"], 0.05, format="%.2f")
    with f4:
        st.session_state["fallback_second_cut"]=st.checkbox("Enable second-cut hints", value=st.session_state["fallback_second_cut"])

    st.divider()
    st.subheader("LLM (Batch+Cache) Options")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.session_state["skip_short_chars"]=st.number_input("Skip if core ≤ chars", 60, 600, st.session_state["skip_short_chars"], 10)
    with c2:
        st.session_state["batch_group_size"]=st.number_input("Batch group size", 2, 20, st.session_state["batch_group_size"], 1)
    with c3:
        st.session_state["parallel_calls"]=st.number_input("Parallel batches", 1, 8, st.session_state["parallel_calls"], 1)
    with c4:
        st.session_state["max_calls"]=st.number_input("Max batch calls (0=unlimited)", 0, 9999, st.session_state["max_calls"], 1)

    st.session_state["use_llm_summary"]=st.checkbox("Generate summaries with LLM", value=st.session_state["use_llm_summary"])

    # Load file content
    if up is not None:
        try:
            st.session_state["text"]=up.read().decode("utf-8")
            st.session_state["source_file"]=up.name
        except UnicodeDecodeError:
            st.error("Please upload a UTF-8 encoded .txt file.")
            return

    if parse_btn:
        if not st.session_state["text"]:
            st.warning("Please upload a file first.")
            return

        panel = RunPanel()
        try:
            text = st.session_state["text"]; src = st.session_state["source_file"]

            # 1) Parse
            panel.step("1/6 Parse")
            t0 = time.time()
            doc_type = detect_doc_type(text)
            result: ParseResult = parse_document(text, doc_type=doc_type)
            panel.log(f"Parsing finished · {time.time()-t0:.2f}s, doc_type={doc_type}")
            panel.tick("Parse", inc=18)

            # 2) Repair / Validate
            panel.step("2/6 Tree repair & validate")
            t0 = time.time()
            issues_before = validate_tree(result)
            rep_diag = repair_tree(result)
            issues_after = validate_tree(result)
            panel.log(f"Issues before={len(issues_before)} → after={len(issues_after)} · {time.time()-t0:.2f}s")
            panel.tick("Tree repair", inc=12)

            # 3) Chunking
            panel.step("3/6 Chunking")
            t0 = time.time()
            base_law = _normalize_law_name(guess_law_name(text), src or "document.txt")
            chunks, mk_diag = make_chunks(
                result=result, mode="article_only", source_file=src, law_name=base_law,
                include_front_matter=True, include_headnotes=True, include_gap_fallback=True,
                allowed_headnote_levels=["ภาค","ลักษณะ","หมวด","ส่วน","บท"],
                min_headnote_len=24, min_gap_len=24, strict_lossless=st.session_state["strict_lossless"],
                split_long_articles=st.session_state["split_long_articles"],
                split_threshold_chars=st.session_state["split_threshold_chars"],
                tail_merge_min_chars=st.session_state["tail_merge_min_chars"],
                overlap_chars=st.session_state["overlap_chars"],
                soft_cut=True
            )
            mode = "structured"
            panel.log(f"Initial chunks: {len(chunks)} · {time.time()-t0:.2f}s")

            # 3a) Structure-free fallback (only if no articles and single big gap)
            arts = [c for c in chunks if (c.meta or {}).get("type") in ("article","article_pack")]
            gaps  = [c for c in chunks if (c.meta or {}).get("type") == "gap_fallback"]
            if st.session_state["fallback_enable"] and len(arts)==0 and len(gaps)==1:
                panel.log("Structure-free fallback triggered: paragraph+length")
                fallback_chunks, fb_diag = make_generic_chunks(
                    full_text=result.full_text,
                    law_name=base_law or "",
                    source_file=src or "",
                    target_chars=st.session_state["fallback_target_chars"],
                    min_ratio=st.session_state["fallback_min_ratio"],
                    overlap_chars=st.session_state["overlap_chars"],
                    tail_merge_min_chars=st.session_state["tail_merge_min_chars"],
                    enable_second_cut=st.session_state["fallback_second_cut"]
                )
                chunks = fallback_chunks
                mode = "fallback"
                if isinstance(mk_diag, dict):
                    mk_diag["generic_fallback"] = fb_diag
                panel.log(f"Fallback chunks: {len(chunks)}")

            # 3b) Tail sweep
            tail_threshold = st.session_state["tail_merge_min_chars"]
            sweep_diag = merge_small_trailing_parts(chunks, full_text=result.full_text, max_tail_chars=tail_threshold)
            if isinstance(mk_diag, dict):
                mk_diag["micro_sweeper_tail"] = {"max_tail_chars": tail_threshold, **sweep_diag}
            panel.log(f"Micro sweeper · merged={sweep_diag.get('merged_count',0)}")

            # 3c) Enumeration boundary snap (pairwise) + label reconciliation
            enum_diag = normalize_enumeration_boundaries(chunks, full_text=result.full_text, reconcile_labels=True)
            panel.log(f"Anchor snap · pairs={enum_diag['pairs_seen']}, hits={enum_diag['hits']}, label_fixes={enum_diag['label_mismatch_fixed']}")

            # 3d) Zero-length & gaps cleanup
            zero_diag = fix_zero_length_and_gaps(chunks, full_text=result.full_text)
            panel.log(f"Zero-length cleanup · removed={zero_diag['removed_zero_span']}, fixed_next_start={zero_diag['fixed_next_start']}")

            # hard-cap check (diagnostic)
            hard_cap = st.session_state["split_threshold_chars"] + 2*st.session_state["overlap_chars"] + 120
            cap_viol = [{"label": (c.meta or {}).get("section_label",""), "len": (int(c.span_end) - int(c.span_start))}
                        for c in chunks if (int(c.span_end) - int(c.span_start)) > hard_cap]
            if isinstance(mk_diag, dict):
                mk_diag["enum_boundary"] = enum_diag
                mk_diag["zero_cleanup"] = {k:v for k,v in zero_diag.items() if k!="removed_labels"}
                mk_diag["hard_cap"] = {"cap_chars": hard_cap, "violations": cap_viol, "count": len(cap_viol)}

            arts = [c for c in chunks if (c.meta or {}).get("type") in ("article","article_pack")]
            panel.log(f"Total chunks now: {len(chunks)} (article-like {len(arts)})")
            panel.tick("Chunking", inc=20)

            # 4) LLM summaries (batch+cache+skip)
            llm_errors: List[str] = []
            cache: Dict[str, Any] = st.session_state["llm_cache"]
            skip_short = st.session_state["skip_short_chars"]

            stats = {
                "model": None, "batches": 0, "sections_total": len(arts),
                "sections_cache_hits": 0, "sections_local_skip": 0, "sections_new_calls": 0,
                "usage_prompt_tokens": 0, "usage_completion_tokens": 0, "usage_total_tokens": 0,
                "elapsed_ms_sum": 0, "est_cost_usd": 0.0,
            }

            if st.session_state["use_llm_summary"] and len(arts)>0:
                panel.step("4/6 LLM batch summarization")
                pending = []
                for idx, c in enumerate(arts, 1):
                    cs,ce = (c.meta or {}).get("core_span",[int(c.span_start), int(c.span_end)])
                    core = result.full_text[int(cs):int(ce)] if (isinstance(cs,int) and isinstance(ce,int) and ce>cs) else c.text
                    core = (core or "").strip()
                    if not core:
                        continue
                    key = _sha1(core)
                    if len(core) <= skip_short:
                        rec = build_summary_record(
                            law_name=base_law or "",
                            section_label=(c.meta or {}).get("section_label",""),
                            breadcrumbs=getattr(c,"breadcrumbs",[]) or [],
                            span=(int(c.span_start), int(c.span_end)),
                            llm_text=core,
                            brief_max_len=180
                        )
                        (c.meta or {}).update({
                            "brief":rec["brief"],"topics":rec["topics"],"negations":rec["negations"],
                            "summary_text_raw":rec["summary_text_raw"],"quality":rec["quality"],"cache":"local-skip"
                        })
                        stats["sections_local_skip"] += 1
                        continue
                    if key in cache:
                        rec = cache[key]["rec"]
                        (c.meta or {}).update({
                            "brief":rec["brief"],"topics":rec["topics"],"negations":rec["negations"],
                            "summary_text_raw":cache[key]["text"],"quality":rec["quality"],"cache":"hit"
                        })
                        stats["sections_cache_hits"] += 1
                        continue
                    pending.append((f"A{idx:04d}", c, core))

                stats["sections_new_calls"] = len(pending)
                panel.log(f"Skip/Cache applied · cache {stats['sections_cache_hits']} / local-skip {stats['sections_local_skip']} / pending {len(pending)}")

                total_batches = math.ceil(len(pending) / st.session_state["batch_group_size"])
                if st.session_state["max_calls"] and total_batches > st.session_state["max_calls"]:
                    total_batches = st.session_state["max_calls"]
                panel.log(f"Planned batch calls: {total_batches}")

                def _run_batch(batch_items):
                    sections = [{"id": bid, "title": (ch.meta or {}).get("section_label",""),
                                 "breadcrumbs": " / ".join(getattr(ch,"breadcrumbs",[]) or []), "text": core}
                                for bid, ch, core in batch_items]
                    return call_openai_batch(sections, max_tokens=1200, temperature=0.2)

                from concurrent.futures import ThreadPoolExecutor, as_completed
                workers = max(1, min(8, st.session_state["parallel_calls"]))
                group = max(2, min(20, st.session_state["batch_group_size"]))
                done_calls = 0
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures=[]; cursor=0; future_to_batch={}
                    while cursor < len(pending) and (st.session_state["max_calls"]==0 or done_calls < st.session_state["max_calls"]):
                        batch = pending[cursor: cursor+group]
                        fut = ex.submit(_run_batch, batch)
                        futures.append(fut); future_to_batch[fut] = batch
                        done_calls += 1; cursor += group
                    for fut in as_completed(futures):
                        r = fut.result()
                        if not r.get("ok"):
                            llm_errors.append(str(r.get("error"))); continue
                        id_map = r.get("map", {})
                        stats["batches"] += 1
                        stats["elapsed_ms_sum"] += int(r.get("ms", 0))
                        if r.get("usage"):
                            u = r["usage"]
                            stats["usage_prompt_tokens"] += int(u.get("prompt_tokens",0))
                            stats["usage_completion_tokens"] += int(u.get("completion_tokens",0))
                            stats["usage_total_tokens"] += int(u.get("total_tokens",0))
                        if r.get("est_cost_usd") is not None:
                            stats["est_cost_usd"] += float(r["est_cost_usd"] or 0.0)
                        if r.get("model"): stats["model"]=r["model"]

                        for bid, ch, core in future_to_batch[fut]:
                            if bid in id_map:
                                txt = id_map[bid]
                                rec = build_summary_record(
                                    law_name=base_law or "",
                                    section_label=(ch.meta or {}).get("section_label",""),
                                    breadcrumbs=getattr(ch,"breadcrumbs",[]) or [],
                                    span=(int(ch.span_start), int(ch.span_end)),
                                    llm_text=txt,
                                    brief_max_len=180
                                )
                                (ch.meta or {}).update({
                                    "brief":rec["brief"],"topics":rec["topics"],"negations":rec["negations"],
                                    "summary_text_raw":rec["summary_text_raw"],"quality":rec["quality"],"cache":"miss"
                                })
                                cache[_sha1(core)] = {"text": txt, "rec": rec}

                st.session_state["llm_cache"]=cache
                panel.tick("LLM summarize", inc=30)
            else:
                if len(arts)==0:
                    panel.log("LLM summarization skipped: no article-like chunks")
                else:
                    panel.log("LLM summarization disabled")
                # keep empty stats structure

            st.session_state["llm_errors"] = llm_errors

            # Derived stats
            def _safe_div(a, b): return (a / b) if (b and b != 0) else 0.0
            derived = {
                "avg_tokens_per_section": round(_safe_div(stats["usage_total_tokens"], max(1, stats.get("sections_new_calls",0))), 2),
                "avg_ms_per_batch": round(_safe_div(stats["elapsed_ms_sum"], max(1, stats.get("batches",0))), 2),
                "sections_per_second": round(_safe_div(stats.get("sections_new_calls",0), stats["elapsed_ms_sum"]/1000.0 if stats["elapsed_ms_sum"] else 0), 3) if stats.get("sections_new_calls",0) else 0.0,
                "effective_unit_cost_per_1k": round(_safe_div(stats["est_cost_usd"], (stats["usage_total_tokens"]/1000.0) if stats["usage_total_tokens"] else 0), 6) if stats["usage_total_tokens"] else None,
                "pricing_source": "derived_effective"
            }
            stats.update({"derived": derived})

            # 5) Report / export
            panel.step("5/6 Report & export")
            cov=_coverage(chunks, len(result.full_text))
            report_str = make_debug_report(
                parse_result=result, chunks=chunks, source_file=src, law_name=base_law or "",
                run_config={
                    "mode": mode,  # <— structured | fallback
                    "strict_lossless":st.session_state["strict_lossless"],
                    "split_long_articles":st.session_state["split_long_articles"],
                    "split_threshold_chars":st.session_state["split_threshold_chars"],
                    "tail_merge_min_chars":st.session_state["tail_merge_min_chars"],
                    "overlap_chars":st.session_state["overlap_chars"],
                    "fallback_enable":st.session_state["fallback_enable"],
                    "fallback_target_chars":st.session_state["fallback_target_chars"],
                    "fallback_min_ratio":st.session_state["fallback_min_ratio"],
                    "fallback_second_cut":st.session_state["fallback_second_cut"],
                    "show_front_matter":st.session_state["bracket_front_matter"],
                    "show_headnotes":st.session_state["bracket_headnotes"],
                    "use_llm_summary":st.session_state["use_llm_summary"],
                    "batch_group_size":st.session_state["batch_group_size"],
                    "parallel_calls":st.session_state["parallel_calls"],
                    "skip_short_chars":st.session_state["skip_short_chars"],
                },
                debug={
                    "tree_repair":{"issues_before":len(issues_before),"issues_after":len(issues_after),**(rep_diag or {})},
                    "make_chunks_diag":mk_diag or {},
                    "coverage_calc":{"coverage":cov},
                    "llm":{"errors":llm_errors, "cache_size":len(st.session_state["llm_cache"]), "stats":stats}
                }
            )

            jsonl_rich = to_jsonl_rich_meta(chunks)
            jsonl_flat = to_jsonl_compat_flat(chunks, parse_result=result)

            st.session_state.update({
                "parsed":True, "result":result, "chunks":chunks,
                "doc_type":getattr(result, "doc_type", None) or "unknown", "law_name":base_law or "",
                "issues":issues_after, "report_json_str":report_str,
                "chunks_jsonl_rich":jsonl_rich, "chunks_jsonl_flat":jsonl_flat
            })
            panel.tick("Report/export", inc=10)

            # 6) Render
            panel.step("6/6 Render")
            panel.finalize(ok=True)

        except Exception as e:
            panel.log(f"❌ Error: {type(e).__name__}: {e}")
            panel.finalize(ok=False)
            raise

    # ----------------------- Display & downloads -----------------------
    if st.session_state["parsed"]:
        cov=_coverage(st.session_state["chunks"], len(st.session_state["result"].full_text))
        arts=[c for c in st.session_state["chunks"] if (c.meta or {}).get("type") in ("article","article_pack")]
        st.write(
            f"**File:** {st.session_state['source_file']}  |  "
            f"**doc_type:** {st.session_state.get('doc_type','unknown')}  |  "
            f"**law_name:** {st.session_state.get('law_name') or 'N/A'}  |  "
            f"**chunks:** {len(st.session_state['chunks'])} (article-like {len(arts)})  |  "
            f"**coverage:** {cov:.6f}"
        )

        if st.session_state["llm_errors"]:
            st.markdown(f'<span class="badge-warn">LLM failures {len(st.session_state["llm_errors"])} items</span>', unsafe_allow_html=True)
            with st.expander("Show LLM failure reasons"):
                for e in st.session_state["llm_errors"]:
                    st.code(e)

        base = _basename_no_ext(st.session_state['source_file'] or "document")
        rich_bytes = st.session_state["chunks_jsonl_rich"].encode("utf-8")
        flat_bytes = st.session_state["chunks_jsonl_flat"].encode("utf-8")
        report_bytes = st.session_state["report_json_str"].encode("utf-8")

        # One-click ZIP
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{base}_chunks_rich.jsonl", rich_bytes)
            zf.writestr(f"{base}_chunks_compat.jsonl", flat_bytes)
            zf.writestr(f"{base}_REPORT.json", report_bytes)
        zip_buf.seek(0)

        st.markdown('<div class="dlbar">', unsafe_allow_html=True)
        st.download_button("Download All (ZIP)", data=zip_buf, file_name=f"{base}_ALL.zip", mime="application/zip", key="dl-all-zip")
        # Optional: still offer individual downloads
        st.download_button("Download — Rich JSONL", rich_bytes, file_name=f"{base}_chunks_rich.jsonl", mime="application/json", key="dl-jsonl-rich")
        st.download_button("Download — Compat JSONL", flat_bytes, file_name=f"{base}_chunks_compat.jsonl", mime="application/json", key="dl-jsonl-flat")
        st.download_button("Download — REPORT.json", report_bytes, file_name=f"{base}_REPORT.json", mime="application/json", key="dl-report-json")
        st.markdown('</div>', unsafe_allow_html=True)

        html = render_with_overlap(
            text=st.session_state["result"].full_text,
            chunks=st.session_state["chunks"],
            include_front=st.session_state["bracket_front_matter"],
            include_head=st.session_state["bracket_headnotes"],
        )
        st.markdown('<div class="docwrap">'+html+'</div>', unsafe_allow_html=True)
    else:
        st.info("Upload a file and click **Parse**.")
        st.caption("Use the sidebar to follow each step's progress and logs.")

if __name__ == "__main__":
    main()
