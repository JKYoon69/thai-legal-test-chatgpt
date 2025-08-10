# -*- coding: utf-8 -*-
import time, json
from typing import List, Dict, Any
import streamlit as st

from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks, guess_law_name, repair_tree
from parser_core.schema import ParseResult, Chunk
from exporters.writers import to_jsonl, make_debug_report
from llm_adapter import LLMRouter

APP_TITLE = "Thai Legal Preprocessor — LLM + Overlap Visual (v2)"

# ───────── CSS ───────── #
def _inject_css():
    st.markdown("""
    <style>
      .block-container { max-width: 1400px !important; padding-top: .75rem; }
      .docwrap { border:1px solid rgba(107,114,128,.25); border-radius:10px; padding:16px; }
      .doc { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
             font-size:13.5px; line-height:1.6; position:relative; }

      /* 전체 청크(연두 중괄호) */
      .chunk-wrap { position:relative; padding:0 12px; border-radius:6px; margin:0 1px; }
      .chunk-wrap::before, .chunk-wrap::after {
        content:"{"; position:absolute; top:-1px; bottom:-1px; width:10px;
        color:#a3e635; opacity:.90; font-weight:800;
      }
      .chunk-wrap::before { left:0; }
      .chunk-wrap::after  { right:0; transform:scaleX(-1); }

      /* 오버랩(연핑크 배경 + 얇은 핑크 중괄호) */
      .overlap { background: rgba(244,114,182,.22); }   /* pink-400 @ 22% */
      .overlap-mark { color:#f472b6; font-weight:800; }

      .close-tail { color:#a3e635; font-weight:800; }
      .dlbar { display:flex; gap:10px; align-items:center; margin:.6rem 0 .6rem; flex-wrap:wrap; }
      .parse-line { margin-top:.25rem; margin-bottom:.5rem; }
      .parse-line button { background:#22c55e !important; color:#fff !important; border:0 !important;
                           padding:.75rem 1.2rem !important; font-weight:800 !important; font-size:16px !important;
                           border-radius:10px !important; width:100%; }
      .badge-warn { display:inline-block; padding:2px 8px; border-radius:8px; background:#fee2e2; color:#b91c1c; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

def _ensure_state():
    defaults = {
        "text":None, "source_file":None, "parsed":False,
        "doc_type":None, "law_name":None, "result":None, "chunks":[], "issues":[],
        "strict_lossless":True, "split_long_articles":True,
        "split_threshold_chars":1500, "tail_merge_min_chars":200, "overlap_chars":200,
        "number_scope":"article만", "bracket_front_matter":True,
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

# ───────── 원문 렌더(오버랩 시각화) ───────── #
def render_with_overlap(text: str, chunks: List[Chunk], *, number_scope="article만", bracket_front_matter=True) -> str:
    N=len(text)
    if number_scope=="전체 청크":
        units=[c for c in chunks if c.meta.get("type") in ("front_matter","article","headnote")]
    else:
        units=[c for c in chunks if c.meta.get("type")=="article"]
        if bracket_front_matter:
            fms=[c for c in chunks if c.meta.get("type")=="front_matter"]
            units=(fms+units) if fms else units
    units.sort(key=lambda c:(c.span_start,c.span_end))

    parts=[]; cur=0; idx=0
    for c in units:
        s,e=int(c.span_start),int(c.span_end)
        if s<0 or e>N or e<=s: continue
        if s>cur: parts.append(_esc(text[cur:s]))

        # 코어스팬(문서 절대좌표) – 없거나 이상하면 전체를 코어로 간주
        cs,ce = c.meta.get("core_span",[s,e])
        if not (isinstance(cs,int) and isinstance(ce,int) and s<=cs<=ce<=e):
            cs,ce=s,e

        parts.append(f'<span class="chunk-wrap" title="{_esc(c.meta.get("section_label","chunk"))}">')

        # 앞 오버랩
        if cs>s:
            parts.append('<span class="overlap-mark">{</span>')
            parts.append(f'<span class="overlap">{_esc(text[s:cs])}</span>')
        # 코어
        parts.append(_esc(text[cs:ce]))
        # 뒤 오버랩
        if e>ce:
            parts.append(f'<span class="overlap">{_esc(text[ce:e])}</span>')
            parts.append('<span class="overlap-mark">}</span>')

        parts.append("</span>")  # chunk-wrap 닫기

        idx+=1
        parts.append(f' <span class="close-tail">}} chunk {idx:04d}</span><br/>')
        cur=e
    if cur<N: parts.append(_esc(text[cur:N]))
    return '<div class="doc">'+"".join(parts)+"</div>"

# ───────── Main ───────── #
def main():
    _inject_css(); _ensure_state()
    st.title(APP_TITLE)
    st.caption("연두 중괄호=청크, 연핑크 배경=오버랩(문맥). LLM은 gpt-4.1-mini-2025-04-14 → gemini-2.5-flash → gpt-5 순서.")

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
    with o3: st.session_state["bracket_front_matter"]=st.checkbox("Front matter도 표시", value=st.session_state["bracket_front_matter"])

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

        text = st.session_state["text"]; src = st.session_state["source_file"]

        # 1) parse
        doc_type = detect_doc_type(text)
        result: ParseResult = parse_document(text, doc_type=doc_type)
        issues_before = validate_tree(result); rep_diag = repair_tree(result); issues_after = validate_tree(result)

        # 2) chunks
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

        # 3) LLM
        llm_log={"law":{}, "desc":{}}
        final_doc_type = result.doc_type or "unknown"
        final_law = base_law or ""
        llm_errors: List[str] = []

        if st.session_state["use_llm_law"]:
            router = LLMRouter(
                primary_model="gpt-4.1-mini-2025-04-14",
                fallback1_model="gemini-2.5-flash",
                fallback2_model="gpt-5",
            )
            obj, diag = router.lawname_doctype(text[:1200])
            llm_log["law"]={"diag":diag,"output":obj}
            for e in diag.get("errors", []): llm_errors.append(f'{e.get("provider")}/{e.get("model")}: {e.get("error")}')
            if obj and obj.get("confidence",0)>=0.75:
                final_doc_type = obj.get("doc_type") or final_doc_type
                final_law = obj.get("law_name") or final_law

        if st.session_state["use_llm_desc"]:
            router = LLMRouter(
                primary_model="gpt-4.1-mini-2025-04-14",
                fallback1_model="gemini-2.5-flash",
                fallback2_model="gpt-5",
            )
            items=[]
            for c in chunks:
                if c.meta.get("type")!="article": continue
                cs,ce = c.meta.get("core_span",[c.span_start, c.span_end])
                core = result.full_text[cs:ce] if (isinstance(cs,int) and isinstance(ce,int) and ce>cs) else c.text
                items.append((core[:1200], c.meta.get("section_label",""), c.breadcrumbs or []))
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

        st.session_state["llm_errors"] = llm_errors

        # 4) REPORT (오버랩 샘플 10개 포함)
        cov=_coverage(chunks, len(result.full_text))
        # 오버랩이 실제 있는 청크들 샘플
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
                "use_llm_law":st.session_state["use_llm_law"], "use_llm_desc":st.session_state["use_llm_desc"],
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

    # ───── 표시 & 다운로드 ─────
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

        # LLM 에러 요약 배지
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
            number_scope=st.session_state["number_scope"],
            bracket_front_matter=st.session_state["bracket_front_matter"],
        )
        st.markdown('<div class="docwrap">'+html+'</div>', unsafe_allow_html=True)
    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
