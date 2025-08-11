# -*- coding: utf-8 -*-
"""
Thai Legal Preprocessor — RAG-ready (lossless + debug)

안전한 변경점 요약
- 기존 구조 문서(หมวด/ส่วน/มาตรา 등 존재): 기존 파이프라인 그대로 동작.
- 구조 신호 거의 없음: 자동 '구조-프리(fallback) 청킹' 발동(단락/길이 기반 분할).
- 번호머리(예: (16)/ข้อ/มาตรา/불릿) 경계 스냅 + 커버리지 틈 방지 유지.
- 0-길이 청크 자동 정리.
- UI/겹침표시/LLM 토큰·비용 REPORT 동일.
"""
import os, sys, re, time, json, math
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

# ---------- 유틸 ----------
def _esc(s:str)->str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _coverage(chunks: List, total_len: int) -> float:
    ivs = sorted([[int(c.span_start), int(c.span_end)] for c in chunks], key=lambda x:x[0])
    merged=[]
    for s,e in ivs:
        if not merged or s>merged[-1][1]: merged.append([s,e])
        else: merged[-1][1]=max(merged[-1][1], e)
    covered=sum(e-s for s,e in merged)
    return (covered/total_len) if total_len else 0.0

# ---------- Fallback Chunk (구조-프리) ----------
class SimpleChunk:
    """Duck-typed Chunk: writers/exporters가 접근하는 속성만 구현"""
    def __init__(self, text: str, span_start: int, span_end: int, meta: Dict[str,Any], breadcrumbs: Optional[List[str]]=None):
        self.text = text
        self.span_start = span_start
        self.span_end = span_end
        self.meta = meta or {}
        self.breadcrumbs = breadcrumbs or []

def _paragraph_breaks(text: str) -> List[int]:
    """
    단락 경계 후보: 빈 줄(연속 개행) 기준. 없으면 한 줄 개행도 후보로.
    반환: 절대 오프셋 리스트(코어 종료 후보).
    """
    N = len(text)
    brks = set()
    for m in re.finditer(r'(?:\r?\n[ \t\f\v]*\r?\n)+', text):
        brks.add(m.end())
    # 단락이 전혀 없다면, 한 줄 개행도 후보로
    if not brks:
        for m in re.finditer(r'\r?\n', text):
            brks.add(m.end())
    brks.add(N)
    return sorted(brks)

def make_generic_chunks(full_text: str, *, law_name: str, source_file: str,
                        target_chars: int, min_chars: int, overlap_chars: int,
                        tail_merge_min_chars: int) -> Tuple[List[SimpleChunk], Dict[str,Any]]:
    """
    구조가 없는 문서를 길이 기반으로 안전 분할.
    - 우선 단락 경계를 최대한 활용
    - 없으면 하드 컷(최대 target_chars) + 좌/우 오버랩
    - 마지막 조각이 너무 짧으면 앞 조각과 병합
    """
    N = len(full_text)
    breaks = _paragraph_breaks(full_text)
    cores = []
    pos = 0

    while pos < N:
        # 후보 경계: [pos+min_chars, pos+target_chars] 구간 내 가장 먼 경계
        lo = pos + min_chars
        hi = min(N, pos + target_chars)
        cand = [b for b in breaks if lo <= b <= hi]
        if cand:
            cut = max(cand)
        else:
            # 여유 범위(hi+200)까지 확장해서라도 경계에 맞춰 자르기
            ext_hi = min(N, hi + 200)
            cand2 = [b for b in breaks if hi < b <= ext_hi]
            cut = (min(N, pos + target_chars) if not cand2 else min(max(cand2), N))
        if cut <= pos:
            cut = min(N, pos + target_chars)
            if cut <= pos:  # 안전장치
                break
        cores.append((pos, cut))
        pos = cut

    # tail 병합
    if len(cores) >= 2 and (cores[-1][1] - cores[-1][0]) < tail_merge_min_chars:
        cores[-2] = (cores[-2][0], cores[-1][1])
        cores.pop()

    chunks: List[SimpleChunk] = []
    for i,(cs,ce) in enumerate(cores, start=1):
        # 오버랩 적용
        s = max(0, cs - (overlap_chars if i>1 else 0))
        e = min(N, ce + (overlap_chars if i < len(cores) else 0))
        meta = {
            "type": "article",  # 다운스트림 호환을 위해 article로 둠
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
        "tail_merge_min_chars": tail_merge_min_chars
    }
    return chunks, diag

# ---------- 번호머리 앵커 스냅 + 커버리지 복구 ----------
# 타이 숫자 범위: \u0E50-\u0E59 (๐–๙)
ENUM_PATTERNS = [
    ("digit", re.compile(r'^\s*[\(\[\{]?\s*(?:\d{1,4}|[\u0E50-\u0E59]{1,4})\s*[\)\]\}]?\s*')),
    ("lawterm", re.compile(r'^\s*(?:ข้อ|มาตรา)\s*(?:\d{1,4}|[\u0E50-\u0E59]{1,4})\s*')),
    ("bullet", re.compile(r'^\s*(?:•|·|-|—|–)\s+')),
]

def _line_start(text: str, pos: int) -> int:
    lb = text.rfind("\n", 0, pos)
    return 0 if lb < 0 else lb + 1

def _detect_enum_anchor_at_chunk_start(text: str, s_next: int, lookahead: int = 80) -> Optional[Tuple[int,int,str]]:
    """
    다음 청크 시작 s_next 기준 '줄머리'에서 번호머리/불릿 패턴을 찾는다.
    반환: (enum_start_abs, enum_end_abs, kind) 또는 None
    """
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

def normalize_enumeration_boundaries(chunks, full_text: str) -> Dict[str, Any]:
    """
    번호머리/불릿을 경계 앵커로 보고, '청크 쌍' 단위로 보정한다.
    - prev.span_end = enum_start (번호머리 앞에서 종료)
    - next.span_start = enum_start (틈 제거)
    - next.core_start = enum_start (번호머리를 코어에 포함)
    - 이동량 과도(>120자)이면 안전상 미적용
    """
    diag = {
        "pairs_seen": 0, "hits": 0, "cut_only": 0, "shift_only": 0, "both": 0,
        "pattern_hits": {"digit":0,"lawterm":0,"bullet":0},
        "moved_chars_hist": [], "adjusted_next_start": 0, "core_moved_to_enum": 0, "samples": []
    }
    if not chunks: return diag
    # 대상 정렬
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

        # 1) 이전 청크 오른쪽 절단 (번호머리 제외)
        if int(prev.span_end) > enum_s >= int(prev.span_start):
            prev.span_end = enum_s
            pcs, pce = (prev.meta or {}).get("core_span", [int(prev.span_start), int(prev.span_end)])
            if pce > prev.span_end: pce = prev.span_end
            if pcs < prev.span_start: pcs = prev.span_start
            prev.meta["core_span"] = [int(pcs), int(pce)]
            prev.meta["overlap_right"] = max(0, int(prev.span_end) - int(pce))
            _refresh_text(prev, full_text)
            applied_cut=True

        # 2) 다음 청크 시작을 enum_s로 당김 (틈 제거)
        if enum_s < int(cur.span_start):
            cur.span_start = enum_s
            _refresh_text(cur, full_text)
            diag["adjusted_next_start"] += 1
            applied_next_start=True

        # 3) 다음 청크 core 시작을 enum_s로 이동(번호머리 포함)
        cs, ce = (cur.meta or {}).get("core_span", [int(cur.span_start), int(cur.span_end)])
        cs_new = enum_s
        if cs_new < int(cs):
            cur.meta["core_span"] = [int(cs_new), int(ce)]
            cur.meta["overlap_left"] = max(0, int(cs_new) - int(cur.span_start))
            applied_shift=True
            diag["core_moved_to_enum"] += 1

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
            if len(diag["samples"]) < 5:
                diag["samples"].append({
                    "next_section": (cur.meta or {}).get("section_label",""),
                    "enum": full_text[enum_s:enum_e],
                    "cut_prev_to": int(prev.span_end),
                    "new_next_start": int(cur.span_start),
                    "new_core_start": int(cur.meta["core_span"][0])
                })
    return diag

def fix_zero_length_and_gaps(chunks: List, full_text: str) -> Dict[str, Any]:
    """
    스냅/절단 후 발생한 0-길이 청크 정리 및 틈 방지.
    - span_end <= span_start 인 청크를 제거
    - 제거 직전, 다음 청크가 존재하고 next.span_start > this.span_start 이면 next.span_start를 끌어당겨 공백 흡수
    """
    diag = {"removed_zero_span":0, "fixed_next_start":0, "removed_labels":[]}
    # 정렬 보장
    chunks.sort(key=lambda c:(int(c.span_start), int(c.span_end)))
    i=0
    while i < len(chunks):
        c = chunks[i]
        s,e = int(c.span_start), int(c.span_end)
        if e <= s:
            # 다음 청크로 틈 메우기
            if i+1 < len(chunks):
                nxt = chunks[i+1]
                if int(nxt.span_start) > s:
                    nxt.span_start = s
                    _refresh_text(nxt, full_text)
                    # overlap_left 재계산
                    cs, ce = (nxt.meta or {}).get("core_span", [int(nxt.span_start), int(nxt.span_end)])
                    nxt.meta["overlap_left"] = max(0, int(cs) - int(nxt.span_start))
                    diag["fixed_next_start"] += 1
            diag["removed_zero_span"] += 1
            diag["removed_labels"].append((c.meta or {}).get("section_label",""))
            del chunks[i]
            continue
        i+=1
    return diag

# ---------- 시각화 ----------
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
        # 왼쪽 겹침은 일반 텍스트
        parts.append(_esc(text[s:cs]))
        # 코어
        parts.append(_esc(text[cs:ce]))
        # 오른쪽 겹침만 핑크 + 표식은 핑크 '뒤'
        if e>ce:
            over_txt = text[ce:e]
            parts.append(f'<span class="overlap">{_esc(over_txt)}</span>')
            if (e - ce) > 4:
                parts.append('<span class="overlap-mark">}</span>')
        parts.append("</span>")

        idx+=1
        brief = (c.meta or {}).get("brief")
        if not brief:
            # 간단 브리프 (코어 앞 180자)
            brief = (text[cs: min(ce, cs+180)] or "").replace("\n"," ").strip()
        tail = f' }} chunk {idx:04d}'
        if brief:
            tail += f' — { _esc(brief) }'
        parts.append(f' <span class="close-tail">{tail}</span><br/>')
        cur=e
    if cur<N: parts.append(_esc(text[cur:N]))
    return '<div class="doc">'+"".join(parts)+"</div>"

# ---------- 앱 상태 ----------
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

            # === 구조-프리 Fallback: 기사/헤드가 전혀 없고 gap만 있을 때만 작동 ===
            arts = [c for c in chunks if (c.meta or {}).get("type") in ("article","article_pack")]
            gaps  = [c for c in chunks if (c.meta or {}).get("type") == "gap_fallback"]
            if len(arts)==0 and len(gaps)==1:
                panel.log("구조-프리 Fallback 작동: 단락/길이 기반 청킹으로 대체")
                fallback_chunks, fb_diag = make_generic_chunks(
                    full_text=result.full_text,
                    law_name=base_law or "",
                    source_file=src or "",
                    target_chars=st.session_state["split_threshold_chars"],
                    min_chars=max(300, int(st.session_state["split_threshold_chars"]*0.45)),
                    overlap_chars=st.session_state["overlap_chars"],
                    tail_merge_min_chars=st.session_state["tail_merge_min_chars"]
                )
                chunks = fallback_chunks
                if isinstance(mk_diag, dict):
                    mk_diag["generic_fallback"] = fb_diag
                panel.log(f"Fallback 생성 청크 {len(chunks)}개")

            # 3.5) 꼬리 병합(일반 문서에서도 적용)
            tail_threshold = st.session_state["tail_merge_min_chars"]
            sweep_diag = merge_small_trailing_parts(chunks, full_text=result.full_text, max_tail_chars=tail_threshold)
            if isinstance(mk_diag, dict):
                mk_diag["micro_sweeper_tail"] = {"max_tail_chars": tail_threshold, **sweep_diag}
            panel.log(f"마이크로 스위퍼 적용 · merged={sweep_diag.get('merged_count',0)}")

            # 3.6) 번호머리 경계 스냅(쌍 단위)
            enum_diag = normalize_enumeration_boundaries(chunks, full_text=result.full_text)
            if isinstance(mk_diag, dict):
                mk_diag["enum_boundary"] = enum_diag
            panel.log(f"경계 스냅 · pairs={enum_diag['pairs_seen']}, hits={enum_diag['hits']}")

            # 3.7) 0-길이/틈 정리
            zero_diag = fix_zero_length_and_gaps(chunks, full_text=result.full_text)
            if isinstance(mk_diag, dict):
                mk_diag["zero_cleanup"] = {k:v for k,v in zero_diag.items() if k!="removed_labels"}
            panel.log(f"0-길이 정리 · removed={zero_diag['removed_zero_span']}, fixed_next_start={zero_diag['fixed_next_start']}")

            # 하드캡 진단
            hard_cap = st.session_state["split_threshold_chars"] + 2*st.session_state["overlap_chars"] + 120
            cap_viol = [{"label": (c.meta or {}).get("section_label",""), "len": (int(c.span_end) - int(c.span_start))}
                        for c in chunks if (int(c.span_end) - int(c.span_start)) > hard_cap]
            if isinstance(mk_diag, dict):
                mk_diag["hard_cap"] = {"cap_chars": hard_cap, "violations": cap_viol, "count": len(cap_viol)}

            arts = [c for c in chunks if (c.meta or {}).get("type") in ("article","article_pack")]
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

            if st.session_state["use_llm_summary"] and len(arts)>0:
                panel.step("4/6 LLM 요약(배치)")
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
                panel.log(f"스킵/캐시 적용 · cache {stats['sections_cache_hits']} / local-skip {stats['sections_local_skip']} / 대기 {len(pending)}")

                total_batches = math.ceil(len(pending) / st.session_state["batch_group_size"])
                if st.session_state["max_calls"] and total_batches > st.session_state["max_calls"]:
                    total_batches = st.session_state["max_calls"]
                panel.log(f"배치 호출 예정: {total_batches}개")

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
                panel.tick("LLM 요약", inc=30)
            else:
                if len(arts)==0:
                    panel.log("LLM 요약: article 없음 → 생략")
                else:
                    panel.log("LLM 요약: 비활성화 (로컬 규칙만)")
                stats = {
                    "model": None, "batches": 0,
                    "sections_total": len(arts),
                    "sections_cache_hits": 0,
                    "sections_local_skip": len(arts) if len(arts)>0 else 0,
                    "sections_new_calls": 0,
                    "usage_prompt_tokens": 0,
                    "usage_completion_tokens": 0,
                    "usage_total_tokens": 0,
                    "elapsed_ms_sum": 0,
                    "est_cost_usd": 0.0,
                }

            st.session_state["llm_errors"] = llm_errors

            # 파생지표
            def _safe_div(a, b): return (a / b) if (b and b != 0) else 0.0
            derived = {
                "avg_tokens_per_section": round(_safe_div(stats["usage_total_tokens"], max(1, stats["sections_new_calls"])), 2),
                "avg_ms_per_batch": round(_safe_div(stats["elapsed_ms_sum"], max(1, stats["batches"])), 2),
                "sections_per_second": round(_safe_div(stats["sections_new_calls"], stats["elapsed_ms_sum"]/1000.0 if stats["elapsed_ms_sum"] else 0), 3),
                "effective_unit_cost_per_1k": round(_safe_div(stats["est_cost_usd"], (stats["usage_total_tokens"]/1000.0) if stats["usage_total_tokens"] else 0), 6) if stats["usage_total_tokens"] else None,
                "pricing_source": "derived_effective"
            }
            stats.update({"derived": derived})

            # 5) REPORT/저장
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
        arts=[c for c in st.session_state["chunks"] if (c.meta or {}).get("type") in ("article","article_pack")]
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
