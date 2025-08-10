# -*- coding: utf-8 -*-
import time, json
from typing import List
import streamlit as st

from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks, guess_law_name, repair_tree
from parser_core.schema import ParseResult, Chunk
from exporters.writers import to_jsonl, make_debug_report
from llm_adapter import LLMRouter

APP_TITLE = "Thai Legal Preprocessor — RAG-ready (lossless + debug)"

# ---------- CSS ----------
def _inject_css():
    st.markdown("""
    <style>
      .block-container { max-width: 1400px !important; padding-top: .6rem; }
      .docwrap { border:1px solid rgba(107,114,128,.25); border-radius:10px; padding:16px; }
      .doc { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
             font-size:13.5px; line-height:1.6; position:relative; }
      /* 청크 전체(연두 괄호) */
      .chunk-wrap { position:relative; padding:0 12px; border-radius:6px; margin:0 1px; }
      .chunk-wrap::before, .chunk-wrap::after {
        content:"{"; position:absolute; top:-1px; bottom:-1px; width:10px;
        color:#a3e635; opacity:.90; font-weight:800;
      }
      .chunk-wrap::before { left:0; }
      .chunk-wrap::after  { right:0; transform:scaleX(-1); }
      /* 오버랩(연핑크 배경 + 얇은 핑크 괄호) */
      .overlap { background: rgba(244,114,182,.22); }
      .overlap-mark { color:#f472b6; font-weight:800; }
      .close-tail { color:#a3e635; font-weight:800; }
      .dlbar { display:flex; gap:10px; align-items:center; margin:.6rem 0 .6rem; flex-wrap:wrap; }
      .parse-line { margin:.25rem 0 .5rem; }
      .parse-line button { background:#22c55e !important; color:#fff !important; border:0 !important;
                           padding:.75rem 1.2rem !important; font-weight:800 !important; font-size:16px !important;
                           border-radius:10px !important; width:100%; }
      .badge-warn { display:inline-block; padding:2px 8px; border-radius:8px; background:#fee2e2; color:#b91c1c; font-weight:700; }
      .muted { color:#9ca3af; font-size:12px; }
    </style>
    """, unsafe_allow_html=True)

def _ensure_state():
    defaults = {
        "text":None, "source_file":None, "parsed":False,
        "doc_type":None, "law_name":None, "result":None, "chunks":[], "issues":[],
        "strict_lossless":True, "split_long_articles":True,
        "split_threshold_chars":1500, "tail_merge_min_chars":200, "overlap_chars":200,
        "bracket_front_matter":True, "bracket_headnotes":True,
        "use_llm_law":True, "use_llm_desc":True,
        "report_json_str":"", "chunks_jsonl_str":"", "llm_errors":[]}
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

# ---------- 원문 렌더 ----------
def render_with_overlap(text: str, chunks: List[Chunk], *, include_front=True, include_head=True) -> str:
    N=len(text)
    units=[]
    if include_front:
        units += [c for c in chunks if c.meta.get("type")=="front_matter"]
    if include_head:
        units += [c for c in chunks if c.meta.get("type")=="headnote"]
    units += [c for c in chunks if c.meta.get("type")=="article"]
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
        if cs>s:
            parts.append('<span class="overlap-mark">{</span>')
            parts.append(f'<span class="overlap">{_esc(text[s:cs])}</span>')
        parts.append(_esc(text[cs:ce]))
        if e>ce:
            parts.append(f'<span class="overlap">{_esc(text[ce:e])}</span>')
            parts.append('<span class="overlap-mark">}</span>')
        parts.append("</span>")
        idx+=1
        parts.append(f' <span class="close-tail">}} chunk {idx:04d}</span><br/>')
        cur=e
    if cur<N: parts.append(_esc(text[cur:N]))
    return '<div class="doc">'+"".join(parts)+"</div>"

# ---------- 유틸: 사이드바 상태 패널 ----------
class RunPanel:
    def __init__(self):
        with st.sidebar:
            st.header("진행 상황")
            self.status = st.status("대기 중...", expanded=True)
            self.prog = st.progress(0, text="대기 중")
            self.lines = st.container()
        self.t0 = time.time()
        self.total_weight = 100  # 가중치 총합(퍼센트)
        self.done = 0

    def _update_prog(self, label, inc=None, set_to=None):
        if set_to is not None:
            self.done = max(0, min(self.total_weight, set_to))
        elif inc is not None:
            self.done = max(0, min(self.total_weight, self.done + inc))
        pct = int(self.done)
        self.prog.progress(pct, text=f"{label} · {pct}%")

    def step(self, label, state="running"):
        self.status.update(label=label, state=state)

    def log(self, text):
        with self.lines:
            st.markdown(f"- {text}")

    def done_step(self, label, inc_weight):
        self._update_prog(label, inc=inc_weight)

    def finalize(self, ok=True):
        elapsed = time.time() - self.t0
        self.status.update(label=f"완료 · {elapsed:.1f}s", state=("complete" if ok else "error"))
        self._update_prog("완료", set_to=100)

# ---------- Main ----------
def main():
    _inject_css(); _ensure_state()
    st.title(APP_TITLE)

    up = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])
    with st.container():
        st.markdown('<div class="parse-line">', unsafe_allow_html=True)
        run = st.button("파싱", key="parse_btn_top"); st.markdown('</div>', unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    with c1: st.session_state["split_threshold_chars"]=st.number_input("분할 임계값(문자)",600,6000,st.session_state["split_threshold_chars"],100)
    with c2: st.session_state["tail_merge_min_chars"]=st.number_input("tail 병합 최소 길이(문자)",0,600,st.session_state["tail_merge_min_chars"],10)
    with c3: st.session_state["overlap_chars"]=st.number_input("오버랩(문맥) 길이",0,800,st.session_state["overlap_chars"],25)

    o1,o2,o3=st.columns(3)
    with o1: st.session_state["strict_lossless"]=st.checkbox("Strict 무손실", value=st.session_state["strict_lossless"])
    with o2: st.session_state["split_long_articles"]=st.checkbox("롱 조문 분할", value=st.session_state["split_long_articles"])
    with o3:
        st.session_state["bracket_front_matter"]=st.checkbox("Front matter도 표시", value=st.session_state["bracket_front_matter"])
        st.session_state["bracket_headnotes"]=st.checkbox("Headnote(ส่วน/หมวด 등) 표시", value=st.session_state["bracket_headnotes"])

    l1,l2=st.columns(2)
    with l1: st.session_state["use_llm_law"]=st.checkbox("LLM: 법령명/유형 보정", value=st.session_state["use_llm_law"])
    with l2: st.session_state["use_llm_desc"]=st.checkbox("LLM: 설명자 생성", value=st.session_state["use_llm_desc"])

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
        ok = True
        try:
            text = st.session_state["text"]; src = st.session_state["source_file"]

            # 1) parse (20)
            panel.step("1/6 파싱 시작")
            t0 = time.time()
            doc_type = detect_doc_type(text)
            result: ParseResult = parse_document(text, doc_type=doc_type)
            panel.log(f"파싱 완료 · {time.time()-t0:.2f}s, doc_type={doc_type}")
            panel.done_step("파싱", inc_weight=20)

            # 2) repair/validate (10)
            panel.step("2/6 트리 수복/검증")
            t0 = time.time()
            issues_before = validate_tree(result)
            rep_diag = repair_tree(result)
            issues_after = validate_tree(result)
            panel.log(f"수복 전 이슈={len(issues_before)} → 후={len(issues_after)} · {time.time()-t0:.2f}s")
            panel.done_step("트리 수복", inc_weight=10)

            # 3) make chunks (20)
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
            panel.log(f"청크 {len(chunks)}개 생성 · {time.time()-t0:.2f}s")
            panel.done_step("청킹", inc_weight=20)

            # 4) LLM law meta (10)
            final_doc_type = result.doc_type or "unknown"
            final_law = base_law or ""
            llm_log={"law":{}, "desc":{}}
            llm_errors: List[str] = []

            if st.session_state["use_llm_law"]:
                panel.step("4/6 LLM 메타(법령명/유형)")
                t0 = time.time()
                router = LLMRouter(
                    primary_model="gpt-4.1-mini-2025-04-14",
                    fallback1_model="gemini-2.5-flash",
                    fallback2_model="gpt-5",
                )
                obj, diag = router.lawname_doctype(text[:1600])
                llm_log["law"]={"diag":diag,"output":obj}
                for e in diag.get("errors", []): llm_errors.append(f'{e.get("provider")}/{e.get("model")}: {e.get("error")}')
                if obj and obj.get("confidence",0)>=0.75:
                    final_doc_type = obj.get("doc_type") or final_doc_type
                    final_law = obj.get("law_name") or final_law
                panel.log(f"LLM 메타 완료 · {time.time()-t0:.2f}s")
            else:
                panel.log("LLM 메타: 비활성화")

            panel.done_step("LLM 메타", inc_weight=10)

            # 5) LLM descriptors (30) — per-article sub progress
            if st.session_state["use_llm_desc"]:
                arts = [c for c in chunks if c.meta.get("type")=="article"]
                total = max(1, len(arts))
                per = 30 / total
                panel.step(f"5/6 LLM 설명자 생성 (기사 {len(arts)}개)")
                router = LLMRouter(
                    primary_model="gpt-4.1-mini-2025-04-14",
                    fallback1_model="gemini-2.5-flash",
                    fallback2_model="gpt-5",
                )
                items=[]
                for c in arts:
                    cs,ce = c.meta.get("core_span",[c.span_start, c.span_end])
                    core = result.full_text[cs:ce] if (isinstance(cs,int) and isinstance(ce,int) and ce>cs) else c.text
                    items.append((core[:1600], c.meta.get("section_label",""), c.breadcrumbs or []))
                # 배치 호출이지만, 진행률을 보여주기 위해 한 번에 결과 받아 후처리
                descs, dlog = router.describe_chunks_batch(items)
                llm_log["desc"]=dlog
                for r in dlog.get("routes",[]):
                    for e in r.get("diag",{}).get("errors",[]): llm_errors.append(f'{e.get("provider")}/{e.get("model")}: {e.get("error")}')
                j=0
                for c in chunks:
                    if c.meta.get("type")!="article": continue
                    obj = descs[j] if j<len(descs) else None; j+=1
                    if obj:
                        c.meta["brief"]=obj.get("brief",""); c.meta["topics"]=obj.get("topics",[]); c.meta["negations"]=obj.get("negations",[])
                    panel.done_step("LLM 설명자", inc_weight=per)
                panel.log(f"LLM 설명자 완료 · calls={dlog.get('calls',0)}, ok={dlog.get('ok',0)}, fail={dlog.get('fail',0)}")
            else:
                panel.log("LLM 설명자: 비활성화")
                panel.done_step("LLM 설명자", inc_weight=30)

            st.session_state["llm_errors"] = llm_errors

            # 6) REPORT/RENDER (10)
            panel.step("6/6 리포트/렌더")
            cov=_coverage(chunks, len(result.full_text))
            samples=[]
            for c in chunks:
                s,e=int(c.span_start),int(c.span_end)
                cs,ce = c.meta.get("core_span",[s,e])
                if isinstance(cs,int) and isinstance(ce,int) and (cs>s or ce<e):
                    samples.append({
                        "section": c.meta.get("section_label",""),
                        "span":[s,e], "core_span":[cs,ce],
                        "uid": c.meta.get("section_uid","")
                    })
                    if len(samples)>=10: break

            report_str = make_debug_report(
                parse_result=result, chunks=chunks, source_file=src, law_name=final_law or "",
                run_config={
                    "strict_lossless":st.session_state["strict_lossless"],
                    "split_long_articles":st.session_state["split_long_articles"],
                    "split_threshold_chars":st.session_state["split_threshold_chars"],
                    "tail_merge_min_chars":st.session_state["tail_merge_min_chars"],
                    "overlap_chars":st.session_state["overlap_chars"],
                    "bracket_front_matter":st.session_state["bracket_front_matter"],
                    "bracket_headnotes":st.session_state["bracket_headnotes"],
                    "use_llm_law":st.session_state["use_llm_law"],
                    "use_llm_desc":st.session_state["use_llm_desc"],
                    "primary_llm_model":"gpt-4.1-mini-2025-04-14"
                },
                debug={
                    "tree_repair":{"issues_before":len(issues_before),"issues_after":len(issues_after),**rep_diag},
                    "make_chunks_diag":{**(mk_diag or {}), "overlap_samples":samples},
                    "coverage_calc":{"coverage":cov},
                    "llm":llm_log
                }
            )
            chunks_str = to_jsonl(chunks)

            st.session_state.update({
                "parsed":True, "result":result, "chunks":chunks,
                "doc_type":final_doc_type, "law_name":final_law,
                "issues":issues_after, "report_json_str":report_str, "chunks_jsonl_str":chunks_str
            })

            panel.done_step("리포트/렌더", inc_weight=10)
            panel.finalize(ok=True)

        except Exception as e:
            panel.log(f"❌ 오류: {type(e).__name__}: {e}")
            panel.finalize(ok=False)
            raise

    # ---------- 표시/다운로드 ----------
    if st.session_state["parsed"]:
        cov=_coverage(st.session_state["chunks"], len(st.session_state["result"].full_text))
        arts=[c for c in st.session_state["chunks"] if c.meta.get("type")=="article"]
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
        st.download_button("JSONL 다운로드", st.session_state["chunks_jsonl_str"].encode("utf-8"),
                           file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_chunks.jsonl",
                           mime="application/json", key="dl-jsonl-bottom")
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
        st.caption("사이드바의 ‘진행 상황’ 패널에서 단계별 진행률을 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
