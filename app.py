# -*- coding: utf-8 -*-
import io
import json
from typing import List, Optional

import streamlit as st

from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks, guess_law_name
from parser_core.schema import ParseResult, Node, Chunk
from exporters.writers import to_jsonl, make_zip_bundle, make_debug_report

APP_TITLE = "Thai Legal Preprocessor — Hybrid Multi-Modal RAG"

# ---------- UI Helpers ----------
def _inject_css():
    st.markdown(
        """
        <style>
          /* make page wider */
          .block-container { max-width: 1400px !important; padding-top: 1rem; }
          /* monospace for raw preview */
          .code-like { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: pre-wrap; }
          /* highlight chip */
          mark { background: #fff3bf; padding: 0 .15rem; border-radius: .2rem; }
          /* tree pane */
          .tree-pane { font-size: .95rem; line-height: 1.25rem; }
          .tree-item { margin: .15rem 0; }
          .tree-label { cursor: pointer; }
          .badge { background: #e7f5ff; border: 1px solid #d0ebff; padding: .1rem .4rem; border-radius: .5rem; font-size: .72rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_tree(nodes: List[Node], path: Optional[List[str]] = None, key_prefix: str = "") -> Optional[str]:
    """
    Renders a simple expandable tree. Returns the selected node_id (if any).
    """
    selected = None
    for i, n in enumerate(nodes):
        with st.expander(f"{n.label or ''} {n.num or ''}  "
                         f"{'· ' if (n.label or n.num) else ''}"
                         f"<span class='badge'>L{n.level}</span>",
                         expanded=False):
            st.markdown(f"<div class='tree-pane'><div class='tree-item'><b>{n.label or ''} {n.num or ''}</b></div></div>", unsafe_allow_html=True)
            if st.button("보기 / 하이라이트", key=f"{key_prefix}-sel-{n.node_id}"):
                selected = n.node_id
            if n.children:
                inner = render_tree(n.children, path=[n.label, n.num], key_prefix=f"{key_prefix}-{i}")
                if inner:
                    selected = inner
    return selected

def highlight_text(full_text: str, spans: List[tuple]) -> str:
    """
    Given spans [(start, end), ...], insert <mark> to highlight.
    Spans must be non-overlapping & sorted by start.
    """
    if not spans:
        return f"<div class='code-like'>{full_text}</div>"
    html = []
    last = 0
    for (s, e) in spans:
        s = max(0, min(s, len(full_text)))
        e = max(0, min(e, len(full_text)))
        if s > last:
            html.append(full_text[last:s])
        html.append(f"<mark>{full_text[s:e]}</mark>")
        last = e
    if last < len(full_text):
        html.append(full_text[last:])
    return f"<div class='code-like'>{''.join(html)}</div>"

# ---------- App ----------
def main():
    _inject_css()
    st.title(APP_TITLE)
    st.caption("Upload a Thai legal .txt (UTF-8). It will parse → build hierarchy → chunk → export for hybrid RAG (pgroonga + pgvector).")

    uploaded = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])
    mode = st.radio(
        "청크 모드",
        options=["article_only", "article±1"],
        index=1,
        help="조문(มาตรา/ข้อ) 단위로만 자를지, 앞뒤 조문 하나씩 가볍게 합칠지 선택",
    )
    show_raw = st.checkbox("원문(raw) 보기", value=False)
    st.divider()

    if not uploaded:
        st.info("샘플을 쓰려면 좌측에 파일을 올리세요.")
        return

    # Read text
    raw_bytes = uploaded.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        st.error("파일 인코딩을 UTF-8로 저장해주세요.")
        return

    source_file = uploaded.name

    # Detect doc type & parse
    doc_type = detect_doc_type(text)
    result: ParseResult = parse_document(text, doc_type=doc_type)

    # Law name (heuristic from top lines)
    law_name = guess_law_name(text)

    # Validation + chunks
    issues = validate_tree(result)
    chunks: List[Chunk] = make_chunks(
        result=result,
        mode=mode,
        source_file=source_file,
        law_name=law_name,
    )

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("문서 트리")
        sel_id = render_tree(result.root.children)
        st.caption(f"front_matter 포함 총 노드: {len(result.all_nodes)}· leaf(조문/항) 청크 수: {len(chunks)}")
        if issues:
            with st.expander("검증 리포트(요약)", expanded=False):
                for it in issues[:100]:
                    st.write("• " + it)

        st.subheader("내보내기")
        jsonl_bytes = to_jsonl(chunks).encode("utf-8")
        st.download_button(
            "JSONL(임베딩용) 다운로드",
            data=jsonl_bytes,
            file_name=f"{source_file.rsplit('.',1)[0]}_chunks.jsonl",
            mime="application/json",
        )
        # zip bundle + debug report
        zip_bytes = make_zip_bundle(
            source_text=text,
            parse_result=result,
            chunks=chunks,
            source_file=source_file,
            law_name=law_name,
        )
        st.download_button(
            "검증 번들(ZIP) 다운로드",
            data=zip_bytes.getvalue(),
            file_name=f"{source_file.rsplit('.',1)[0]}_bundle.zip",
            mime="application/zip",
        )

    with col2:
        st.subheader("본문 미리보기")
        if show_raw:
            st.markdown(f"<div class='code-like'>{text}</div>", unsafe_allow_html=True)
        else:
            # If a node selected in tree, highlight its span
            spans = []
            if sel_id:
                node = result.node_map.get(sel_id)
                if node:
                    spans = [(node.span_start, node.span_end)]
            html = highlight_text(text, sorted(spans, key=lambda x: x[0]))
            st.markdown(html, unsafe_allow_html=True)

        st.caption(
            f"메타: doc_type={doc_type}, law_name={law_name or 'N/A'}, file={source_file}, mode={mode}"
        )


if __name__ == "__main__":
    main()
