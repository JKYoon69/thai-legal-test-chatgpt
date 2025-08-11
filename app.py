# -*- coding: utf-8 -*-
import time, json, math, os, sys, re
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st

# --- 경로 보정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 파서/후처리
from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import (
    validate_tree, make_chunks, guess_law_name, repair_tree,
    merge_small_trailing_parts,
)
from parser_core.schema import ParseResult, Chunk

# 익스포트 (writers 위치 호환)
try:
    from writers import to_jsonl_rich_meta, to_jsonl_compat_flat, make_debug_report
except ModuleNotFoundError:
    from exporters.writers import to_jsonl_rich_meta, to_jsonl_compat_flat, make_debug_report

# LLM (요약 배치/캐시)
from llm_clients import call_openai_batch
from extractors import build_summary_record

APP_TITLE = "Thai Legal Preprocessor — RAG-ready (lossless + debug)"

# ---------- 스타일 ----------
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
      .parse-line { margin:.25rem 0 .6rem; }
      .parse-line button { background:#22c55e !important; color:#fff !important; border:0 !important;
                           padding:.75rem 1.2rem !important; font-weight:800 !important; font-size:16px !important;
                           border-radius:10px !important; width:100%; }
      .badge-warn { display:inline-block; padding:2px 8px; border-radius:8px; background:#fee2e2; color:#b91c1c; font-weight:700; }
      .muted { color:#9ca3af; font-size:12px; }
      .kv { font-size:12.5px; color:#93c5fd; }
    </style>
    """, unsafe_allow_html=True)

def _ensure_state():
    defaults = {
        "text":None, "source_file":None, "parsed":False,
        "doc_type":None, "law_name":None, "result":None, "chunks":[], "issues":[],
        "strict_lossless":True, "split_long_articles":True,
        "split_threshold_chars":1500, "tail_merge_min_chars":200, "overlap_chars":200,
        "bracket_front_matter":True, "bracket_headnotes":True,
        "use_llm_summary":True, "llm_errors":[], "report_json_str":"", "chunks_jsonl_rich":"", "chunks_jsonl_flat":"",
        # LLM 성능/비용 옵션
        "skip_short_chars":180,
        "batch_group_size":8,
        "parallel_calls":3,
        "max_calls":0,
        "llm_cache":{}
    }
    for k,v in defaults.items(): st.session_state.setdefault(k,v)

def _coverage(chunks: List[Chunk], total_len: int) -> float:
    ivs = sorted([[c.span_start, c.span_end] for c in chunks], key=lambda x:x[0])
    merged=[]
    for s,e in ivs:
        if not merged or s>merged[-1][1]: merged.append([s,e])
        else: merged[-1][1]=max(merged[-1][1], e)
    covered=sum(e-s for s,e in merged)
    return (covered/total_len) if total_len else 0.0

def _esc(s:str)->str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# ---------- 경계 스냅(번호머리 앵커) ----------
# 타이 숫자 범위: \u0E50-\u0E59 (๐–๙)
ENUM_PATTERNS = [
    re.compile(r'^\s*[\(\[\{]?\s*(?:\d{1,4}|[\u0E50-\u0E59]{1,4})\s*[\)\]\}]?\s*'),  # (16) / 16) / ๑๖)
    re.compile(r'^\s*(?:ข้อ|มาตรา)\s*(?:\d{1,4}|[\u0E50-\u0E59]{1,4})\s*'),           # ข้อ 12 / มาตรา 5
    re.compile(r'^\s*(?:•|·|-|—|–)\s+'),                                             # 불릿
]

def _match_enum_at(text: str, pos: int, lookbehind: int = 12, lookahead: int = 50) -> Optional[Tuple[int,int]]:
    """
    pos 주변(앞 12, 뒤 50)에서 번호머리/불릿을 '선두'로 매칭.
    반환: (abs_start, abs_end) 또는 None
    """
    N = len(text)
    a = max(0, pos - lookbehind)
    b = min(N, pos + lookahead)
    sub = text[a:b]
    for rx in ENUM_PATTERNS:
        m = rx.match(sub)
        if m:
            return (a + m.start(), a + m.end())
    return None

def normalize_enumeration_boundaries(chunks: List[Chunk], full_text: str) -> Dict[str, Any]:
    """
    번호머리/불릿을 경계 앵커로 보고:
    - 다음 청크(core 시작) 앞의 번호머리가 이전 청크 '오른쪽 오버랩'에 끼어있으면 잘라낸다.
    - 다음 청크 core 시작이 번호머리 '뒤'라면, 번호머리 '앞'으로 core 시작을 당겨서 포함한다.
    무손실 유지(coverage 보장), overlap_left/right 재계산.
    """
    diag = {"cuts_prev_overlap": 0, "shifted_core_start": 0, "samples": []}
    if not chunks: return diag

    # 정렬 보장
    chunks.sort(key=lambda c:(int(c.span_start), int(c.span_end)))

    for i in range(len(chunks)):
        c = chunks[i]
        ctype = (c.meta or {}).get("type") or "article"
        if ctype not in ("article", "article_pack"):  # 핵심 구간만 처리
            continue

        # 현재 core span
        cs, ce = (c.meta or {}).get("core_span", [c.span_start, c.span_end])
        if not (isinstance(cs,int) and isinstance(ce,int) and c.span_start <= cs <= ce <= c.span_end):
            cs, ce = c.span_start, c.span_end

        # cs 주변에 번호머리 존재?
        hit = _match_enum_at(full_text, cs)
        if hit:
            enum_s, enum_e = hit

            # 1) core 시작이 번호머리 '뒤'라면, core 시작을 번호머리 '앞'으로 이동
            if enum_s < cs <= enum_e + 2:  # cs가 번호머리 직후라면 포함시키자
                cs_new = enum_s
                if cs_new < cs:
                    c.meta["core_span"] = [cs_new, ce]
                    # overlap_left 재계산
                    c.meta["overlap_left"] = max(0, cs_new - c.span_start)
                    diag["shifted_core_start"] += 1

            # 2) 이전 청크의 오른쪽 오버랩에 번호머리가 끼어있으면 잘라낸다
            if i > 0:
                prev = chunks[i-1]
                if prev.span_end > enum_s:
                    prev.span_end = enum_s  # 번호머리 앞에서 종료
                    pcs, pce = (prev.meta or {}).get("core_span", [prev.span_start, prev.span_end])
                    if prev.span_end < pce:
                        pce = prev.span_end
                    prev.meta["core_span"] = [pcs, pce]
                    prev.meta["overlap_right"] = max(0, prev.span_end - pce)
                    diag["cuts_prev_overlap"] += 1
                    if len(diag["samples"]) < 5:
                        diag["samples"].append({
                            "next_section": (c.meta or {}).get("section_label",""),
                            "enum": full_text[enum_s:enum_e]
                        })

    return diag

# ---------- 시각화 ----------
# 왼쪽 겹침은 일반 텍스트, 오른쪽 겹침만 핑크. `}` 표식은 핑크 '뒤'에 둔다.
def render_with_overlap(text: str, chunks: List[Chunk], *, include_front=True, include_head=True) -> str:
    N=len(text)
    units=[]
    if include_front:
        units += [c for c in chunks if c.meta.get("type")=="front_matter"]
    if include_head:
        units += [c for c in chunks if c.meta.get("type")=="headnote"]
    units += [c for c in chunks if c.meta.get("type") in ("article","article_pack")]
    units.sort(key=lambda c:(c.span_start,c.span_end))

    parts=[]; cur=0; idx=0
    for c in units:
        s,e=int(c.span_start),int(c.span_end)
        if s<0 or e>N or e<=s: continue
        if s>cur: parts.append(_esc(text[cur:s]))

        cs,ce = c.meta.get("core_span",[s,e])
        if not (isinstance(cs,int) and isinstance(ce,int) and s<=cs<=ce<=e):
            cs,ce=s,e

        parts.append(f'<span class="chunk-wrap" title="{_esc(c.meta.get("section_label","chunk"))}">')
        # 왼쪽 겹침은 일반 텍스트
        parts.append(_esc(text[s:cs]))
        # 코어
        parts.append(_esc(text[cs:ce]))
        # 오른쪽 겹침만 핑크 + 표식은 핑크 '뒤'
        if e>ce:
            parts.append(f'<span class="overlap">{_esc(text[ce:e])}</span>')
            parts.append('<span class="overlap-mark">}</span>')
        parts.append("</span>")

        idx+=1
        brief = (c.meta or {}).get("brief") or _brief_fallback(c, text, 180)
        tail = f' }} chunk {idx:04d}'
        if brief:
            tail += f' — { _esc(brief) }'
        parts.append(f' <span class="close-tail">{tail}</span><br/>')
        cur=e
    if cur<N: parts.append(_esc(text[cur:N]))
    return '<div class="doc">'+"".join(parts)+"</div>"

def _brief_fallback(chunk: Chunk, full_text: str, max_len: int = 180) -> str:
    s, e = chunk.meta.get("core_span",[chunk.span_start, chunk.span_end])
    if not (isinstance(s,int) and isinstance(e,int) and e>s): s,e = chunk.span_start, chunk.span_end
    core = full_text[s:e].strip()
    if not core: core = (chunk.text or "").strip()
    if not core: return ""
    rec = build_summary_record(
        law_name=chunk.meta.get("law_name",""),
        section_label=chunk.meta.get("section_label",""),
        breadcrumbs=chunk.breadcrumbs or [],
        span=(chunk.span_start, chunk.span_end),
        llm_text=core,
        brief_max_len=max_len
    )
    return rec.get("brief","")

class RunPanel:
    def __init__(self):
        with st.sidebar:
            st.header("진행 상황")
            self.status = st.status("대기 중...", expanded=True)
            self.prog = st.progress(0, text="대기 중")
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
        self.status.update(label=f"완료 · {time.time()-self.t0:.1f}s", state=("complete" if ok else "error"))
        self._update("완료", set_to=100)

def _sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---------- 메인 ----------
def main():
    _inject_css(); _ensure_state()
    st.title(APP_TITLE)

    up = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])
    with st.container():
        st.markdown('<div class="parse-line">', unsafe_allow_html=True)
        run = st.button("파싱", key="parse_btn_top"); st.markdown('</div>', unsafe_allow_html=True)

    # 옵션
    r1,r2,r3 = st.columns(3)
    with r1: st.session_state["split_threshold_chars"]=st.number_input("분할 임계값(문자)",600,6000,st.session_state["split_threshold_chars"],100)
    with r2: st.session_state["tail_merge_min_chars"]=st.number_input("tail 병합 최소 길이(문자)",0,600,st.session_state["tail_merge_min_chars"],10)
    with r3: st.session_state["overlap_chars"]=st.number_input("오버랩(문맥) 길이",0,800,st.session_state["overlap_chars"],25)

    o1,o2,o3 = st.columns(3)
    with o1: st.session_state["strict_lossless"]=st.checkbox("Strict 무손실", value=st.session_state["strict_lossless"])
    with o2: st.session_state["split_long_articles"]=st.checkbox("롱 조문 분할", value=st.session_state["split_long_articles"])
    with o3:
        st.session_state["bracket_front_matter"]=st.checkbox("Front matter도 표시", value=st.session_state["bracket_front_matter"])
        st.session_state["bracket_headnotes"]=st.checkbox("Headnote(ส่วน/หมวด 등) 표시", value=st.session_state["bracket_headnotes"])

    st.divider()
    st.subheader("LLM 속도/비용 옵션")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.session_state["skip_short_chars"]=st.number_input("LLM 스킵 임계(문자)", 60, 600, st.session_state["skip_short_chars"], 10)
    with c2: st.session_state["batch_group_size"]=st.number_input("배치 그룹 크기(개)", 2, 20, st.session_state["batch_group_size"], 1)
    with c3: st.session_state["parallel_calls"]=st.number_input("동시 배치 호출", 1, 8, st.session_state["parallel_calls"], 1)
    with c4: st.session_state["max_calls"]=st.number_input("최대 배치 호출 수(0=무제한)", 0, 9999, st.session_state["max_calls"], 1)

    st.session_state["use_llm_summary"]=st.checkbox("LLM 요약 생성(배치+캐시)", value=True)

    if up is not None:
        try:
            st.session_state["text"]=up.read().decode("utf-8")
            st.session_state["source_file"]=up.name
        except UnicodeDecodeError:
            st.error("UTF-8로 저장된 .txt를 업로드하세요."); return

    if run:
        if not st.session_state["text"]:
            st.warning("먼저 파일을 업로드하세요."); return

        panel = RunPanel()
        try:
            text = st.session_state["text"]; src = st.session_state["source_file"]

            # 1) 파싱
            panel.step("1/6 파싱")
            t0 = time.time()
            doc_type = detect_doc_type(text)
            result: ParseResult = parse_document(text, doc_type=doc_type)
            panel.log(f"파싱 완료 · {time.time()-t0:.2f}s, doc_type={doc_type}")
            panel.tick("파싱", inc=18)

            # 2) 수복/검증
            panel.step("2/6 트리 수복/검증")
            t0 = time.time()
            issues_before = validate_tree(result)
            rep_diag = repair_tree(result)
            issues_after = validate_tree(result)
            panel.log(f"수복 전 이슈={len(issues_before)} → 후={len(issues_after)} · {time.time()-t0:.2f}s")
            panel.tick("트리 수복", inc=12)

            # 3) 청킹
            panel.step("3/6 청킹")
            t0 = time.time()
            base_law = guess_law_name(text)
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
            panel.log(f"청크 1차 생성 {len(chunks)}개 · {time.time()-t0:.2f}s")

            # 3.5) 꼬리 병합
            tail_threshold = st.session_state["tail_merge_min_chars"]
            sweep_diag = merge_small_trailing_parts(chunks, full_text=result.full_text, max_tail_chars=tail_threshold)
            if isinstance(mk_diag, dict):
                mk_diag["micro_sweeper_tail"] = {"max_tail_chars": tail_threshold, **sweep_diag}
            panel.log(f"마이크로 스위퍼 적용 · merged={sweep_diag['merged_count']}")

            # 3.6) 번호머리 경계 스냅(근본 정리)
            enum_diag = normalize_enumeration_boundaries(chunks, full_text=result.full_text)
            if isinstance(mk_diag, dict):
                mk_diag["enum_boundary"] = enum_diag
            panel.log(f"경계 스냅 · cut_prev_overlap={enum_diag['cuts_prev_overlap']}, shift_core={enum_diag['shifted_core_start']}")

            # 하드캡 진단(분할 임계 + 2*overlap + 120)
            hard_cap = st.session_state["split_threshold_chars"] + 2*st.session_state["overlap_chars"] + 120
            cap_viol = [{"label": (c.meta or {}).get("section_label",""), "len": (c.span_end - c.span_start)}
                        for c in chunks if (c.span_end - c.span_start) > hard_cap]
            if isinstance(mk_diag, dict):
                mk_diag["hard_cap"] = {"cap_chars": hard_cap, "violations": cap_viol, "count": len(cap_viol)}

            arts = [c for c in chunks if c.meta.get("type") in ("article","article_pack")]
            panel.log(f"현재 청크 {len(chunks)}개 (article {len(arts)})")
            panel.tick("청킹", inc=20)

            # 4) LLM 요약 (배치+캐시+스킵) — 사용량/시간 누적 통계
            llm_errors: List[str] = []
            cache: Dict[str, Any] = st.session_state["llm_cache"]
            skip_short = st.session_state["skip_short_chars"]

            stats = {
                "model": None,
                "batches": 0,
                "sections_total": len(arts),
                "sections_cache_hits": 0,
                "sections_local_skip": 0,
                "sections_new_calls": 0,
                "usage_prompt_tokens": 0,
                "usage_completion_tokens": 0,
                "usage_total_tokens": 0,
                "elapsed_ms_sum": 0,
                "est_cost_usd": 0.0,
            }

            if st.session_state["use_llm_summary"]:
                panel.step("4/6 LLM 요약(배치)")
                pending: List[Tuple[str, Chunk, str]] = []
                for idx, c in enumerate(arts, 1):
                    cs,ce = c.meta.get("core_span",[c.span_start, c.span_end])
                    core = result.full_text[cs:ce] if (isinstance(cs,int) and isinstance(ce,int) and ce>cs) else c.text
                    core = (core or "").strip()
                    if not core:
                        continue
                    key = _sha1(core)
                    if len(core) <= skip_short:
                        rec = build_summary_record(
                            law_name=base_law or "",
                            section_label=c.meta.get("section_label",""),
                            breadcrumbs=c.breadcrumbs or [],
                            span=(c.span_start, c.span_end),
                            llm_text=core,
                            brief_max_len=180
                        )
                        c.meta.update({
                            "brief":rec["brief"],"topics":rec["topics"],"negations":rec["negations"],
                            "summary_text_raw":rec["summary_text_raw"],"quality":rec["quality"],"cache":"local-skip"
                        })
                        stats["sections_local_skip"] += 1
                        continue
                    if key in cache:
                        rec = cache[key]["rec"]
                        c.meta.update({
                            "brief":rec["brief"],"topics":rec["topics"],"negations":rec["negations"],
                            "summary_text_raw":cache[key]["text"],"quality":rec["quality"],"cache":"hit"
                        })
                        stats["sections_cache_hits"] += 1
                        continue
                    pending.append((f"A{idx:04d}", c, core))

                stats["sections_new_calls"] = len(pending)
                panel.log(f"스킵/캐시 적용 · cache {stats['sections_cache_hits']} / local-skip {stats['sections_local_skip']} / 대기 {len(pending)}")

                total_batches = math.ceil(len(pending) / st.session_state["batch_group_size"])
                if st.session_state["max_calls"] and total_batches > st.session_state["max_calls"]:
                    total_batches = st.session_state["max_calls"]
                panel.log(f"배치 호출 예정: {total_batches}개")

                def _run_batch(batch_items):
                    sections = []
                    for bid, ch, core in batch_items:
                        sections.append({
                            "id": bid,
                            "title": ch.meta.get("section_label",""),
                            "breadcrumbs": " / ".join(ch.breadcrumbs or []),
                            "text": core
                        })
                    return call_openai_batch(sections, max_tokens=1200, temperature=0.2)

                workers = max(1, min(8, st.session_state["parallel_calls"]))
                group = max(2, min(20, st.session_state["batch_group_size"]))
                done_calls = 0
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures=[]; cursor=0; future_to_batch={}
                    while cursor < len(pending) and (st.session_state["max_calls"]==0 or done_calls < st.session_state["max_calls"]):
                        batch = pending[cursor: cursor+group]
                        fut = ex.submit(_run_batch, batch)
                        futures.append(fut)
                        future_to_batch[fut] = batch
                        done_calls += 1
                        cursor += group
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
                                    section_label=ch.meta.get("section_label",""),
                                    breadcrumbs=ch.breadcrumbs or [],
                                    span=(ch.span_start, ch.span_end),
                                    llm_text=txt,
                                    brief_max_len=180
                                )
                                ch.meta.update({
                                    "brief":rec["brief"],"topics":rec["topics"],"negations":rec["negations"],
                                    "summary_text_raw":rec["summary_text_raw"],"quality":rec["quality"],"cache":"miss"
                                })
                                cache[_sha1(core)] = {"text": txt, "rec": rec}

                st.session_state["llm_cache"]=cache
                panel.tick("LLM 요약", inc=30)
            else:
                panel.log("LLM 요약: 비활성화 (로컬 규칙만)")
                for c in arts:
                    cs,ce = c.meta.get("core_span",[c.span_start, c.span_end])
                    core = result.full_text[cs:ce] if (isinstance(cs,int) and isinstance(ce,int) and ce>cs) else c.text
                    rec = build_summary_record(
                        law_name=base_law or "",
                        section_label=c.meta.get("section_label",""),
                        breadcrumbs=c.breadcrumbs or [],
                        span=(c.span_start, c.span_end),
                        llm_text=core,
                        brief_max_len=180
                    )
                    c.meta.update({
                        "brief":rec["brief"],"topics":rec["topics"],"negations":rec["negations"],
                        "summary_text_raw":rec["summary_text_raw"],"quality":rec["quality"],"cache":"local-only"
                    })
                panel.tick("LLM 요약", inc=30)
                stats = {
                    "model": None, "batches": 0,
                    "sections_total": len(arts),
                    "sections_cache_hits": 0,
                    "sections_local_skip": len(arts),
                    "sections_new_calls": 0,
                    "usage_prompt_tokens": 0,
                    "usage_completion_tokens": 0,
                    "usage_total_tokens": 0,
                    "elapsed_ms_sum": 0,
                    "est_cost_usd": 0.0,
                }

            st.session_state["llm_errors"] = llm_errors

            # 5) REPORT/JSONL
            panel.step("5/6 리포트/저장")
            cov=_coverage(chunks, len(result.full_text))
            report_str = make_debug_report(
                parse_result=result, chunks=chunks, source_file=src, law_name=base_law or "",
                run_config={
                    "strict_lossless":st.session_state["strict_lossless"],
                    "split_long_articles":st.session_state["split_long_articles"],
                    "split_threshold_chars":st.session_state["split_threshold_chars"],
                    "tail_merge_min_chars":st.session_state["tail_merge_min_chars"],
                    "overlap_chars":st.session_state["overlap_chars"],
                    "bracket_front_matter":st.session_state["bracket_front_matter"],
                    "bracket_headnotes":st.session_state["bracket_headnotes"],
                    "use_llm_summary":st.session_state["use_llm_summary"],
                    "batch_group_size":st.session_state["batch_group_size"],
                    "parallel_calls":st.session_state["parallel_calls"],
                    "skip_short_chars":st.session_state["skip_short_chars"],
                },
                debug={
                    "tree_repair":{"issues_before":len(issues_before),"issues_after":len(issues_after),**(rep_diag or {})},
                    "make_chunks_diag":mk_diag or {},
                    "coverage_calc":{"coverage":cov},
                    "llm":{
                        "errors":llm_errors,
                        "cache_size":len(st.session_state["llm_cache"]),
                        "stats":stats
                    }
                }
            )

            jsonl_rich = to_jsonl_rich_meta(chunks)
            jsonl_flat = to_jsonl_compat_flat(chunks, parse_result=result)

            st.session_state.update({
                "parsed":True, "result":result, "chunks":chunks,
                "doc_type":result.doc_type or "unknown", "law_name":base_law or "",
                "issues":issues_after, "report_json_str":report_str,
                "chunks_jsonl_rich":jsonl_rich, "chunks_jsonl_flat":jsonl_flat
            })
            panel.tick("리포트/저장", inc=10)

            # 6) 렌더
            panel.step("6/6 렌더")
            panel.finalize(ok=True)

        except Exception as e:
            panel.log(f"❌ 오류: {type(e).__name__}: {e}")
            panel.finalize(ok=False)
            raise

    # ---------- 표시/다운로드 ----------
    if st.session_state["parsed"]:
        cov=_coverage(st.session_state["chunks"], len(st.session_state["result"].full_text))
        arts=[c for c in st.session_state["chunks"] if c.meta.get("type") in ("article","article_pack")]
        st.write(
            f"**파일:** {st.session_state['source_file']}  |  "
            f"**doc_type:** {st.session_state.get('doc_type','unknown')}  |  "
            f"**law_name:** {st.session_state.get('law_name') or 'N/A'}  |  "
            f"**chunks:** {len(st.session_state['chunks'])} (article {len(arts)})  |  "
            f"**coverage:** {cov:.6f}"
        )

        if st.session_state["llm_errors"]:
            st.markdown(f'<span class="badge-warn">LLM 실패 {len(st.session_state["llm_errors"])}건</span>', unsafe_allow_html=True)
            with st.expander("LLM 실패 사유 보기"):
                for e in st.session_state["llm_errors"]:
                    st.code(e)

        st.markdown('<div class="dlbar">', unsafe_allow_html=True)
        st.download_button("JSONL 다운로드 — Rich Meta", st.session_state["chunks_jsonl_rich"].encode("utf-8"),
                           file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_chunks_rich.jsonl",
                           mime="application/json", key="dl-jsonl-rich")
        st.download_button("JSONL 다운로드 — Compat Flat", st.session_state["chunks_jsonl_flat"].encode("utf-8"),
                           file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_chunks_compat.jsonl",
                           mime="application/json", key="dl-jsonl-flat")
        st.download_button("DEBUG 다운로드 (REPORT.json)", st.session_state["report_json_str"].encode("utf-8"),
                           file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_REPORT.json",
                           mime="application/json", key="dl-report-bottom")
        st.markdown('</div>', unsafe_allow_html=True)

        html = render_with_overlap(
            text=st.session_state["result"].full_text,
            chunks=st.session_state["chunks"],
            include_front=st.session_state["bracket_front_matter"],
            include_head=st.session_state["bracket_headnotes"],
        )
        st.markdown('<div class="docwrap">'+html+'</div>', unsafe_allow_html=True)
    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")
        st.caption("사이드바 ‘진행 상황’ 패널에서 단계별 진행률을 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
