# -*- coding: utf-8 -*-
import time, json
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st

# 파서 파이프라인
from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import (
    validate_tree,
    make_chunks,
    guess_law_name,
    repair_tree,
)
from parser_core.schema import ParseResult, Node, Chunk

# 디버그/출력
from exporters.writers import to_jsonl, make_debug_report

# LLM 어댑터
from llm_adapter import LLMRouter

APP_TITLE = "Thai Legal Preprocessor — LLM-assisted (law_name/doc_type + descriptors)"

# ───────── CSS ───────── #
def _inject_css():
    st.markdown(
        """
        <style>
          .block-container { max-width: 1400px !important; padding-top: .75rem; }
          .docwrap { border: 1px solid rgba(107,114,128,.25); border-radius: 10px; padding: 16px; }
          .doc { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                 font-size: 13.5px; line-height: 1.6; position: relative; }

          .bracket-block { position: relative; padding: 0 12px; border-radius: 6px; margin: 0 1px; }
          .bracket-block::before, .bracket-block::after {
            content: "{"; position: absolute; top: -1px; bottom: -1px; width: 10px;
            color: #a3e635; opacity: .90; font-weight: 800;
          }
          .bracket-block::before { left: 0; }
          .bracket-block::after  { right: 0; transform: scaleX(-1); }

          .close-tail { color: #a3e635; font-weight: 800; }
          .dlbar { display:flex; gap:10px; align-items:center; margin: .6rem 0 .6rem; }
          .parse-line { margin-top: .25rem; margin-bottom: .5rem; }
          .parse-line button { background: #22c55e !important; color: white !important; border: 0 !important;
                               padding: .75rem 1.2rem !important; font-weight: 800 !important; font-size: 16px !important;
                               border-radius: 10px !important; width: 100%; }
        </style>
        """, unsafe_allow_html=True
    )

def _ensure_state():
    defaults = {
        "text": None,
        "source_file": None,
        "parsed": False,
        "doc_type": None,
        "law_name": None,
        "result": None,
        "chunks": [],
        "issues": [],
        # 분할/오버랩 옵션
        "strict_lossless": True,
        "split_long_articles": True,
        "split_threshold_chars": 1500,
        "tail_merge_min_chars": 200,
        "overlap_chars": 200,
        # 브래킷 표시 옵션
        "bracket_mode": "청크",
        "number_scope": "article만",
        "bracket_front_matter": True,
        # LLM 토글
        "use_llm_law": True,
        "use_llm_desc": True,
        # 다운로드 텍스트
        "report_json_str": "",
        "chunks_jsonl_str": "",
        # LLM 로그
        "llm_log": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def _coverage(chunks: List[Chunk], total_len: int) -> float:
    ivs = sorted([[c.span_start, c.span_end] for c in chunks], key=lambda x: x[0])
    merged = []
    for s, e in ivs:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    covered = sum(e - s for s, e in merged)
    return (covered / total_len) if total_len else 0.0

def _html_escape(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# ───────── 표시(브래킷) ───────── #

def _collect_article_leaves(root: Node) -> List[Node]:
    leaves: List[Node] = []
    stack = list(root.children)
    while stack:
        n = stack.pop(0)
        if n.label in ("มาตรา", "ข้อ"):
            leaves.append(n)
        for c in n.children:
            stack.append(c)
    leaves.sort(key=lambda x: x.span_start)
    return leaves

def render_chunk_brackets(text: str, chunks: List[Chunk], *, number_scope="article만", bracket_front_matter=True) -> str:
    N = len(text)
    if number_scope == "전체 청크":
        units = [c for c in chunks if c.meta.get("type") in ("front_matter","article","headnote")]
    else:
        units = [c for c in chunks if c.meta.get("type") == "article"]
        if bracket_front_matter:
            fms = [c for c in chunks if c.meta.get("type") == "front_matter"]
            units = (fms + units) if fms else units
    units.sort(key=lambda c: (c.span_start, c.span_end))

    parts: List[str] = []
    cur = 0
    idx = 0
    for c in units:
        s, e = int(c.span_start), int(c.span_end)
        if s < 0 or e > N or e <= s:
            continue
        if s > cur:
            parts.append(_html_escape(text[cur:s]))
        seg = _html_escape(text[s:e])
        parts.append(f'<span class="bracket-block" title="{_html_escape(c.meta.get("section_label","chunk"))}">{seg}</span>')
        idx += 1
        tail = f' <span class="close-tail">}} chunk {idx:04d}</span><br/>'
        parts.append(tail)
        cur = e
    if cur < N:
        parts.append(_html_escape(text[cur:N]))
    return '<div class="doc">' + "".join(parts) + "</div>"

# ───────── Main ───────── #

def main():
    _inject_css()
    _ensure_state()

    st.title(APP_TITLE)
    st.caption("Upload UTF-8 Thai legal .txt → [파싱] → LLM 보조 메타(선택) → 원문+브래킷")

    uploaded = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])

    with st.container():
        st.markdown('<div class="parse-line">', unsafe_allow_html=True)
        parse_clicked = st.button("파싱", key="parse_btn_top")
        st.markdown('</div>', unsafe_allow_html=True)

    # 옵션(한 줄)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state["split_threshold_chars"] = st.number_input("분할 임계값(문자)", 600, 6000, st.session_state["split_threshold_chars"], 100)
    with col2:
        st.session_state["tail_merge_min_chars"] = st.number_input("tail 병합 최소 길이(문자)", 0, 600, st.session_state["tail_merge_min_chars"], 10)
    with col3:
        st.session_state["overlap_chars"] = st.number_input("오버랩(문맥) 길이", 0, 800, st.session_state["overlap_chars"], 25)

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.session_state["strict_lossless"] = st.checkbox("Strict 무손실(coverage=1.0)", value=st.session_state["strict_lossless"])
    with oc2:
        st.session_state["split_long_articles"] = st.checkbox("롱 조문 분할(문단 경계)", value=st.session_state["split_long_articles"])
    with oc3:
        st.session_state["bracket_front_matter"] = st.checkbox("문서 첫부분도 중괄호(Front matter)", value=st.session_state["bracket_front_matter"])

    # LLM 옵션
    lc1, lc2 = st.columns(2)
    with lc1:
        st.session_state["use_llm_law"] = st.checkbox("LLM: 법령명/문서유형 보정 (fallback only)", value=st.session_state["use_llm_law"])
    with lc2:
        st.session_state["use_llm_desc"] = st.checkbox("LLM: 청크 설명자 생성 (brief/topics/negations)", value=st.session_state["use_llm_desc"])

    if uploaded is not None:
        try:
            st.session_state["text"] = uploaded.read().decode("utf-8")
            st.session_state["source_file"] = uploaded.name
        except UnicodeDecodeError:
            st.error("파일 인코딩을 UTF-8로 저장해주세요.")
            return

    if parse_clicked:
        if not st.session_state["text"]:
            st.warning("먼저 파일을 업로드하세요.")
            return

        text = st.session_state["text"]
        source_file = st.session_state["source_file"]

        # 1) detect + parse
        doc_type = detect_doc_type(text)
        result: ParseResult = parse_document(text, doc_type=doc_type)

        # 2) tree repair
        issues_before = validate_tree(result)
        rep_diag = repair_tree(result)
        issues_after = validate_tree(result)

        # 3) chunks
        base_law_name = guess_law_name(text)
        chunks, mk_diag = make_chunks(
            result=result,
            mode="article_only",
            source_file=source_file,
            law_name=base_law_name,
            include_front_matter=True,
            include_headnotes=True,
            include_gap_fallback=True,
            allowed_headnote_levels=["ภาค","ลักษณะ","หมวด","ส่วน","บท"],
            min_headnote_len=24,
            min_gap_len=24,
            strict_lossless=bool(st.session_state["strict_lossless"]),
            split_long_articles=bool(st.session_state["split_long_articles"]),
            split_threshold_chars=int(st.session_state["split_threshold_chars"]),
            tail_merge_min_chars=int(st.session_state["tail_merge_min_chars"]),
            overlap_chars=int(st.session_state["overlap_chars"]),
            soft_cut=True,
        )

        llm_log: Dict[str, Any] = {"law": {}, "desc": {}}

        # 4) LLM: law_name/doc_type 보정 (fallback only)
        final_law_name = base_law_name or ""
        final_doc_type = result.doc_type or "unknown"
        if st.session_state["use_llm_law"]:
            router = LLMRouter(
                primary_model="gpt-4.1-mini",
                fallback1_model="gemini-2.5-flash",
                fallback2_model="gpt-5",
            )
            header_snippet = text[:1200]
            t0 = time.time()
            law_obj, law_diag = router.lawname_doctype(header_snippet)
            dt = round(time.time() - t0, 3)
            llm_log["law"] = {"diag": law_diag, "latency_s": dt, "output": law_obj}
            if law_obj and law_obj.get("confidence", 0) >= 0.75:
                final_law_name = law_obj.get("law_name") or final_law_name
                final_doc_type = law_obj.get("doc_type") or final_doc_type

        # law/doc_type 적용
        st.session_state["doc_type"] = final_doc_type
        st.session_state["law_name"] = final_law_name

        # 5) LLM: descriptors (article chunks only)
        if st.session_state["use_llm_desc"]:
            router = LLMRouter(
                primary_model="gpt-4.1-mini",
                fallback1_model="gemini-2.5-flash",
                fallback2_model="gpt-5",
            )
            items: List[Tuple[str, str, List[str]]] = []
            for c in chunks:
                if c.meta.get("type") != "article":
                    continue
                # core_span 우선
                cs, ce = c.meta.get("core_span", [c.span_start, c.span_end])
                if isinstance(cs, int) and isinstance(ce, int) and ce > cs:
                    core_text = result.full_text[cs:ce]
                else:
                    core_text = c.text
                # 입력 길이 제한(안전)
                core_text = core_text[:1200]
                items.append((core_text, c.meta.get("section_label",""), c.breadcrumbs or []))

            t0 = time.time()
            desc_list, desc_log = router.describe_chunks_batch(items)
            dt = round(time.time() - t0, 3)
            llm_log["desc"] = {"summary": desc_log, "latency_s_total": dt}

            # 결과 반영 (article 순회 순서대로 매핑)
            j = 0
            for c in chunks:
                if c.meta.get("type") != "article":
                    continue
                obj = desc_list[j] if j < len(desc_list) else None
                j += 1
                if obj:
                    c.meta["brief"] = obj.get("brief", "")
                    c.meta["topics"] = obj.get("topics", [])
                    c.meta["negations"] = obj.get("negations", [])
                    c.meta["desc_confidence"] = str(obj.get("confidence", 0))

        # 커버리지/다운로드
        cov = _coverage(chunks, len(result.full_text))
        report_str = make_debug_report(
            parse_result=result,
            chunks=chunks,
            source_file=source_file,
            law_name=st.session_state["law_name"] or "",
            run_config={
                "strict_lossless": bool(st.session_state["strict_lossless"]),
                "split_long_articles": bool(st.session_state["split_long_articles"]),
                "split_threshold_chars": int(st.session_state["split_threshold_chars"]),
                "tail_merge_min_chars": int(st.session_state["tail_merge_min_chars"]),
                "overlap_chars": int(st.session_state["overlap_chars"]),
                "bracket_mode": st.session_state["bracket_mode"],
                "number_scope": st.session_state["number_scope"],
                "bracket_front_matter": bool(st.session_state["bracket_front_matter"]),
                "use_llm_law": bool(st.session_state["use_llm_law"]),
                "use_llm_desc": bool(st.session_state["use_llm_desc"]),
            },
            debug={
                "tree_repair": {"issues_before": len(issues_before), "issues_after": len(issues_after), **rep_diag},
                "make_chunks_diag": mk_diag or {},
                "coverage_calc": {"coverage": cov},
                "llm": llm_log,
            },
        )
        chunks_str = to_jsonl(chunks)

        st.session_state.update({
            "result": result,
            "chunks": chunks,
            "issues": issues_after,
            "parsed": True,
            "report_json_str": report_str,
            "chunks_jsonl_str": chunks_str,
        })

    # ───── 표시 & 다운로드 ─────
    if st.session_state["parsed"]:
        cov = _coverage(st.session_state["chunks"], len(st.session_state["result"].full_text))
        arts = [c for c in st.session_state["chunks"] if c.meta.get("type") == "article"]

        st.write(
            f"**파일:** {st.session_state['source_file']}  |  "
            f"**doc_type:** {st.session_state.get('doc_type','unknown')}  |  "
            f"**law_name:** {st.session_state.get('law_name') or 'N/A'}  |  "
            f"**chunks:** {len(st.session_state['chunks'])} (article {len(arts)})  |  "
            f"**coverage:** {cov:.6f}"
        )

        st.markdown('<div class="dlbar">', unsafe_allow_html=True)
        st.download_button(
            "JSONL 다운로드 (CHUNKS.jsonl)",
            data=st.session_state["chunks_jsonl_str"].encode("utf-8"),
            file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_chunks.jsonl",
            mime="application/json",
            key="dl-jsonl-bottom",
        )
        st.download_button(
            "DEBUG 다운로드 (REPORT.json)",
            data=st.session_state["report_json_str"].encode("utf-8"),
            file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_REPORT.json",
            mime="application/json",
            key="dl-report-bottom",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # 원문 + 청크 브래킷
        html = render_chunk_brackets(
            text=st.session_state["result"].full_text,
            chunks=st.session_state["chunks"],
            number_scope=st.session_state["number_scope"],
            bracket_front_matter=bool(st.session_state["bracket_front_matter"]),
        )
        st.markdown('<div class="docwrap">' + html + "</div>", unsafe_allow_html=True)
    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
