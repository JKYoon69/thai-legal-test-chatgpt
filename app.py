# -*- coding: utf-8 -*-
import io
import json
from typing import List, Optional, Tuple

import streamlit as st

from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks, guess_law_name
from parser_core.schema import ParseResult, Node, Chunk
from exporters.writers import to_jsonl, make_zip_bundle, make_debug_report

APP_TITLE = "Thai Legal Preprocessor — Hybrid Multi-Modal RAG"

# ───────────────────────────────── UI Helpers ───────────────────────────────── #

def _inject_css():
    st.markdown(
        """
        <style>
          .block-container { max-width: 1400px !important; padding-top: .75rem; }
          .toolbar { position: sticky; top: 0; z-index: 10; padding: .5rem 0 .75rem; 
                     background: var(--background-color); border-bottom: 1px solid rgba(128,128,128,.25);}
          .code-like { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: pre-wrap; }
          mark { background: #fff3bf; padding: 0 .15rem; border-radius: .2rem; }
          .tree-pane { font-size: .95rem; line-height: 1.25rem; }
          .tree-row { display: flex; align-items: center; gap: .5rem; margin: .1rem 0; }
          .tree-label { padding: .1rem .25rem; border-radius: .35rem; }
          .depth-1 .tree-label{ background: rgba(0,0,0,.06); }
          .depth-2 .tree-label{ background: rgba(0,0,0,.05); }
          .depth-3 .tree-label{ background: rgba(0,0,0,.04); }
          .depth-4 .tree-label{ background: rgba(0,0,0,.03); }
          .indent { height: 1px; width: var(--indent, 0px); }
          .badge { background: #e7f5ff; border: 1px solid #d0ebff; padding: .05rem .35rem; border-radius: .5rem; font-size: .72rem; }
          /* dark mode friendly */
          @media (prefers-color-scheme: dark){
            .badge { background: rgba(56, 139, 253,.15); border-color: rgba(56,139,253,.35); }
            .depth-1 .tree-label{ background: rgba(255,255,255,.06); }
            .depth-2 .tree-label{ background: rgba(255,255,255,.05); }
            .depth-3 .tree-label{ background: rgba(255,255,255,.04); }
            .depth-4 .tree-label{ background: rgba(255,255,255,.03); }
          }
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
        "sel_id": None,
        "mode": "article±1",
    }.items():
        st.session_state.setdefault(k, v)

def _flatten_with_depth(nodes: List[Node], depth: int = 0) -> List[Tuple[Node, int]]:
    out: List[Tuple[Node, int]] = []
    for n in nodes:
        out.append((n, depth))
        if n.children:
            out.extend(_flatten_with_depth(n.children, depth + 1))
    return out

def render_tree_and_select(result: ParseResult):
    """Flat tree with indentation + per-row '보기' button. Stores selection in session_state['sel_id']"""
    st.subheader("문서 트리")
    if not result or not result.root.children:
        st.info("트리에 표시할 노드가 없습니다.")
        return

    flat = _flatten_with_depth(result.root.children, depth=0)
    for n, depth in flat:
        # indentation width (px)
        indent_px = 14 * depth
        cols = st.columns([1, 5, 1])
        with cols[0]:
            # visual indent
            st.markdown(f"<div class='indent' style='--indent:{indent_px}px'></div>", unsafe_allow_html=True)
        with cols[1]:
            lbl = f"{(n.label or '')} {('' if not n.num else n.num)}".strip()
            st.markdown(
                f"<div class='tree-pane depth-{min(depth,4)}'><span class='tree-label'>{lbl} "
                f"<span class='badge'>L{n.level}</span></span></div>",
                unsafe_allow_html=True,
            )
        with cols[2]:
            if st.button("보기", key=f"view-{n.node_id}"):
                st.session_state["sel_id"] = n.node_id

    st.caption(f"전체 노드: {len(result.all_nodes)}")

def highlight_text(full_text: str, spans: List[tuple]) -> str:
    """Insert <mark> to highlight non-overlapping spans."""
    if not spans:
        return f"<div class='code-like'>{full_text}</div>"
    html = []
    last = 0
    for (s, e) in sorted(spans, key=lambda x: x[0]):
        s = max(0, min(s, len(full_text)))
        e = max(0, min(e, len(full_text)))
        if s > last:
            html.append(full_text[last:s])
        html.append(f"<mark>{full_text[s:e]}</mark>")
        last = e
    if last < len(full_text):
        html.append(full_text[last:])
    return f"<div class='code-like'>{''.join(html)}</div>"

# ───────────────────────────────── App ───────────────────────────────── #

def main():
    _inject_css()
    _ensure_state()

    st.title(APP_TITLE)
    st.caption("Upload a Thai legal .txt (UTF-8). Click [파싱] to build hierarchy → chunk → export for hybrid RAG (pgroonga + pgvector).")

    # inputs
    uploaded = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])
    st.session_state["mode"] = st.radio(
        "청크 모드",
        options=["article_only", "article±1"],
        index=1 if st.session_state["mode"] == "article±1" else 0,
        help="조문(มาตรา/ข้อ) 단위로만 자를지, 앞뒤 조문 하나씩 가볍게 합칠지 선택",
    )
    show_raw = st.checkbox("원문(raw) 보기", value=False)
    parse_clicked = st.button("파싱")

    # load text to state on upload
    if uploaded is not None:
        try:
            st.session_state["text"] = uploaded.read().decode("utf-8")
            st.session_state["source_file"] = uploaded.name
        except UnicodeDecodeError:
            st.error("파일 인코딩을 UTF-8로 저장해주세요.")
            return

    # Only parse when button clicked
    if parse_clicked:
        if not st.session_state["text"]:
            st.warning("먼저 파일을 업로드하세요.")
            return

        text = st.session_state["text"]
        source_file = st.session_state["source_file"]
        mode = st.session_state["mode"]

        # detect → parse
        doc_type = detect_doc_type(text)
        result: ParseResult = parse_document(text, doc_type=doc_type)

        # metadata and chunks
        law_name = guess_law_name(text)
        issues = validate_tree(result)
        chunks: List[Chunk] = make_chunks(
            result=result,
            mode=mode,
            source_file=source_file,
            law_name=law_name,
        )

        # store
        st.session_state["doc_type"] = doc_type
        st.session_state["law_name"] = law_name
        st.session_state["result"] = result
        st.session_state["chunks"] = chunks
        st.session_state["issues"] = issues
        st.session_state["parsed"] = True
        st.session_state["sel_id"] = None  # reset selection

    # After parse: show toolbar (exports) at the very top (sticky)
    if st.session_state["parsed"]:
        with st.container():
            st.markdown("<div class='toolbar'>", unsafe_allow_html=True)
            tcol1, tcol2, tcol3, _ = st.columns([2, 1.2, 1.2, 3])
            with tcol1:
                st.write(
                    f"**파일:** {st.session_state['source_file']}  |  "
                    f"**doc_type:** {st.session_state['doc_type']}  |  "
                    f"**law_name:** {st.session_state['law_name'] or 'N/A'}  |  "
                    f"**chunks:** {len(st.session_state['chunks'])}"
                )
            with tcol2:
                jsonl_bytes = to_jsonl(st.session_state["chunks"]).encode("utf-8")
                st.download_button(
                    "JSONL 다운로드",
                    data=jsonl_bytes,
                    file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_chunks.jsonl",
                    mime="application/json",
                    key="dl-jsonl-top",
                )
            with tcol3:
                zip_bytes = make_zip_bundle(
                    source_text=st.session_state["text"],
                    parse_result=st.session_state["result"],
                    chunks=st.session_state["chunks"],
                    source_file=st.session_state["source_file"],
                    law_name=st.session_state["law_name"] or "",
                )
                st.download_button(
                    "검증 번들(ZIP)",
                    data=zip_bytes.getvalue(),
                    file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_bundle.zip",
                    mime="application/zip",
                    key="dl-zip-top",
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # columns: tree (indent), preview with highlight
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            render_tree_and_select(st.session_state["result"])
            if st.session_state["issues"]:
                with st.expander("검증 리포트(요약)", expanded=False):
                    for it in st.session_state["issues"][:100]:
                        st.write("• " + it)

        with col2:
            st.subheader("본문 미리보기")
            text = st.session_state["text"]
            if show_raw:
                st.markdown(f"<div class='code-like'>{text}</div>", unsafe_allow_html=True)
            else:
                spans = []
                sel_id = st.session_state.get("sel_id")
                if sel_id:
                    node = st.session_state["result"].node_map.get(sel_id)
                    if node:
                        spans = [(node.span_start, node.span_end)]
                html = highlight_text(text, spans)
                st.markdown(html, unsafe_allow_html=True)

            st.caption(
                f"메타: doc_type={st.session_state['doc_type']}, "
                f"law_name={st.session_state['law_name'] or 'N/A'}, "
                f"file={st.session_state['source_file']}, "
                f"mode={st.session_state['mode']}"
            )
    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
