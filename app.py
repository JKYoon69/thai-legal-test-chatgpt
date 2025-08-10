# -*- coding: utf-8 -*-
import io
import json
import time
from typing import List

import streamlit as st

# ─────────────────────────────────────────────
# Always prefer the package layout (parser_core/*)
# Fallback to flat files only if needed.
# ─────────────────────────────────────────────
try:
    from parser_core.parser import detect_doc_type, parse_document
    from parser_core.postprocess import validate_tree, make_chunks, guess_law_name
    from parser_core.schema import ParseResult, Node, Chunk
    from exporters import writers as wr
    IMPORT_FLAVOR = "parser_core/*"
except Exception:
    try:
        from parser import detect_doc_type, parse_document
        from postprocess import validate_tree, make_chunks, guess_law_name
        from schema import ParseResult, Node, Chunk
        import writers as wr
        IMPORT_FLAVOR = "flat *.py"
    except Exception as e:
        raise ImportError(
            "Cannot import parsing modules. Expected either:\n"
            "  - parser_core/{parser,postprocess,schema}.py + exporters/writers.py\n"
            "  - or flat files {parser,postprocess,schema,writers}.py in the app root.\n"
            f"Original import error: {e}"
        ) from e

APP_TITLE = "Thai Legal Preprocessor — RAG-ready (lossless + debug)"

def _inject_css():
    st.markdown(
        """
        <style>
          .block-container { max-width: 1400px !important; padding-top: .75rem; }
          .toolbar { position: sticky; top: 0; z-index: 10; padding: .5rem 0 .75rem;
                     background: var(--background-color); border-bottom: 1px solid rgba(128,128,128,.25); }
          .code-like { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: pre-wrap; }
          mark { background: #fff3bf; padding: 0 .15rem; border-radius: .2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _ensure_state():
    for k, v in {
        "text": None,
        "source_file": None,
        "parsed": False,
        "doc_type": None,
        "law_name": None,
        "result": None,
        "chunks": [],
        "issues": [],
        "mode": "article_only",
        # lossless / noise-control (also saved into REPORT.run_config)
        "include_front_matter": True,
        "include_headnotes": True,
        "include_gap_fallback": True,
        "allowed_headnote_levels": ["ภาค","ลักษณะ","หมวด","ส่วน","บท"],
        "min_headnote_len": 24,
        "min_gap_len": 24,
        "show_raw": False,
    }.items():
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

def main():
    _inject_css()
    _ensure_state()

    st.title(APP_TITLE)
    st.caption(f"Imports: **{IMPORT_FLAVOR}** · Upload UTF-8 Thai legal .txt → [파싱] → lossless chunks + detailed REPORT.json")

    uploaded = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])

    colA, colB, colC = st.columns(3)
    with colA:
        st.session_state["mode"] = st.selectbox("청크 모드", options=["article_only"], index=0)
    with colB:
        st.session_state["show_raw"] = st.checkbox("원문(raw) 보기", value=st.session_state["show_raw"])
    with colC:
        st.write("")

    st.markdown("**손실 방지 / 노이즈 억제 옵션 (REPORT에 기록됩니다)**")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.session_state["include_front_matter"] = st.checkbox("front matter 포함", value=st.session_state["include_front_matter"])
    with oc2:
        st.session_state["include_headnotes"] = st.checkbox("headnote 포함", value=st.session_state["include_headnotes"])
    with oc3:
        st.session_state["include_gap_fallback"] = st.checkbox("gap-sweeper 포함", value=st.session_state["include_gap_fallback"])
    sc1, sc2 = st.columns(2)
    with sc1:
        st.session_state["min_headnote_len"] = st.number_input("headnote 최소 길이(문자)", min_value=0, max_value=400, value=st.session_state["min_headnote_len"])
    with sc2:
        st.session_state["min_gap_len"] = st.number_input("gap 보강 최소 길이(문자)", min_value=0, max_value=400, value=st.session_state["min_gap_len"])

    parse_clicked = st.button("파싱")

    # 파일 적재
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

        t0 = time.time()
        text = st.session_state["text"]
        source_file = st.session_state["source_file"]

        # 1) detect
        t1 = time.time()
        doc_type = detect_doc_type(text)
        t2 = time.time()

        # 2) parse
        result: ParseResult = parse_document(text, doc_type=doc_type)
        t3 = time.time()

        # 3) validate
        issues = validate_tree(result)
        t4 = time.time()

        # 4) chunks
        law_name = guess_law_name(text)
        chunks: List[Chunk] = make_chunks(
            result=result,
            mode=st.session_state["mode"],
            source_file=source_file,
            law_name=law_name,
            include_front_matter=st.session_state["include_front_matter"],
            include_headnotes=st.session_state["include_headnotes"],
            include_gap_fallback=st.session_state["include_gap_fallback"],
            allowed_headnote_levels=list(st.session_state["allowed_headnote_levels"]),
            min_headnote_len=int(st.session_state["min_headnote_len"]),
            min_gap_len=int(st.session_state["min_gap_len"]),
        )
        t5 = time.time()

        st.session_state.update({
            "doc_type": doc_type,
            "law_name": law_name,
            "result": result,
            "chunks": chunks,
            "issues": issues,
            "parsed": True,
        })

        # REPORT 설정/디버그
        st.session_state["run_config"] = {
            "mode": st.session_state["mode"],
            "include_front_matter": st.session_state["include_front_matter"],
            "include_headnotes": st.session_state["include_headnotes"],
            "include_gap_fallback": st.session_state["include_gap_fallback"],
            "allowed_headnote_levels": list(st.session_state["allowed_headnote_levels"]),
            "min_headnote_len": int(st.session_state["min_headnote_len"]),
            "min_gap_len": int(st.session_state["min_gap_len"]),
            "import_flavor": IMPORT_FLAVOR,
        }
        st.session_state["debug"] = {
            "timings_sec": {
                "detect": round(t2 - t1, 6),
                "parse": round(t3 - t2, 6),
                "validate": round(t4 - t3, 6),
                "make_chunks": round(t5 - t4, 6),
                "total": round(t5 - t0, 6),
            }
        }

    # toolbar / export
    if st.session_state["parsed"]:
        cov = _coverage(st.session_state["chunks"], len(st.session_state["result"].full_text))
        with st.container():
            st.markdown("<div class='toolbar'>", unsafe_allow_html=True)
            c1,c2,c3,_ = st.columns([3,1.2,1.2,3])
            with c1:
                st.write(
                    f"**파일:** {st.session_state['source_file']}  |  "
                    f"**doc_type:** {st.session_state['doc_type']}  |  "
                    f"**law_name:** {st.session_state['law_name'] or 'N/A'}  |  "
                    f"**chunks:** {len(st.session_state['chunks'])}  |  "
                    f"**coverage:** {cov:.6f}"
                )
            with c2:
                jsonl_bytes = wr.to_jsonl(st.session_state["chunks"]).encode("utf-8")
                st.download_button("JSONL 다운로드", data=jsonl_bytes,
                                   file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_chunks.jsonl",
                                   mime="application/json", key="dl-jsonl-top")
            with c3:
                zip_bytes = wr.make_zip_bundle(
                    source_text=st.session_state["text"],
                    parse_result=st.session_state["result"],
                    chunks=st.session_state["chunks"],
                    source_file=st.session_state["source_file"],
                    law_name=st.session_state["law_name"] or "",
                    run_config=st.session_state.get("run_config", {}),
                    debug=st.session_state.get("debug", {}),
                )
                st.download_button("검증 번들(ZIP)",
                                   data=zip_bytes.getvalue(),
                                   file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_bundle.zip",
                                   mime="application/zip", key="dl-zip-top")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state["show_raw"]:
            st.subheader("원문 미리보기")
            st.markdown(f"<div class='code-like'>{st.session_state['text'][:2000]}</div>", unsafe_allow_html=True)
        if st.session_state["issues"]:
            st.caption(f"검증 경고: {len(st.session_state['issues'])}건 (REPORT.json에 상세 기록됨)")
    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
