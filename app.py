# -*- coding: utf-8 -*-
import time
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st

# 파이프라인 모듈 (이미 레포에 있는 parser_core/* 사용)
from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import (
    validate_tree,
    make_chunks,
    guess_law_name,
    repair_tree,  # 트리 수복
)
from parser_core.schema import ParseResult, Chunk
from exporters import writers as wr

APP_TITLE = "Thai Legal Preprocessor — Full Text + Leaf Brackets (최하위 노드만 표시)"

# ───────────────────────────── UI helpers ───────────────────────────── #

def _inject_css():
    st.markdown(
        """
        <style>
          .block-container { max-width: 1400px !important; padding-top: .75rem; }
          .toolbar { position: sticky; top: 0; z-index: 10; padding: .5rem 0 .75rem;
                     background: var(--background-color); border-bottom: 1px solid rgba(128,128,128,.25); }
          .docwrap { border: 1px solid rgba(107,114,128,.25); border-radius: 10px; padding: 16px; }
          .doc { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                 font-size: 13.5px; line-height: 1.6; position: relative; }

          /* 녹색 중괄호(최하위 노드만 감쌈) */
          .bracket-block { position: relative; padding: 0 12px; border-radius: 6px; }
          .bracket-block::before, .bracket-block::after {
            content: "{"; position: absolute; top: -1px; bottom: -1px; width: 10px;
            color: #16a34a; font-weight: 700; opacity: .8;
          }
          .bracket-block::before { left: 0; }
          .bracket-block::after  { right: 0; transform: scaleX(-1); }
          .muted { color:#6b7280; }
          .kpi { display:inline-block; padding:2px 8px; border-radius:12px; background:#f3f4f6; margin-right:6px; }
        </style>
        """,
        unsafe_allow_html=True,
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
        # 파이프라인 옵션(필수만 남김)
        "strict_lossless": True,
        "split_long_articles": True,
        "split_threshold_chars": 1500,
        "tail_merge_min_chars": 200,
        # 추출 범위(리프만 쓰므로 헤드/갭은 굳이 안 감쌈)
        "include_front_matter": True,
        "include_headnotes": True,
        "include_gap_fallback": True,
        "min_headnote_len": 24,
        "min_gap_len": 24,
        "allowed_headnote_levels": ["ภาค","ลักษณะ","หมวด","ส่วน","บท"],
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

# ───────────────────────────── Full-text renderer (최단/안전 버전) ───────────────────────────── #

def render_brackets_only(text: str, chunks: List[Chunk]) -> str:
    """
    원문 전체를 그대로 출력하되,
    제일 하위 노드(=조문/ข้อ, meta.type == 'article') 구간만 녹색 중괄호로 감싼다.
    - 겹침/중복은 무시하고, 좌→우 순서로 차례대로 감쌈
    - headnote/orphan_gap 등은 감싸지 않음(원문 그대로)
    """
    N = len(text)
    # 대상: article 타입만
    arts = [c for c in chunks if c.meta.get("type") == "article"]
    arts.sort(key=lambda c: (c.span_start, c.span_end))

    parts: List[str] = []
    cur = 0
    for c in arts:
        s, e = int(c.span_start), int(c.span_end)
        if s < 0 or e > N or e <= s:
            continue
        # 앞쪽 원문(그대로)
        if s > cur:
            parts.append(_html_escape(text[cur:s]))
        # 본문을 녹색 중괄호로 감쌈
        seg = _html_escape(text[s:e])
        tip = c.meta.get("section_label") or "article"
        bc = " / ".join(c.breadcrumbs or [])
        title = tip if not bc else f"{tip} — {bc}"
        parts.append(f'<span class="bracket-block" title="{_html_escape(title)}">{seg}</span>')
        cur = e
    # 꼬리
    if cur < N:
        parts.append(_html_escape(text[cur:N]))

    return '<div class="doc">' + "".join(parts) + "</div>"

# ───────────────────────────── Main ───────────────────────────── #

def main():
    _inject_css()
    _ensure_state()

    st.title(APP_TITLE)
    st.caption("Upload UTF-8 Thai legal .txt → [파싱] → 원문 전체 표시 + 최하위 노드(조문/ข้อ)만 녹색 중괄호로 감싸기")

    uploaded = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])

    colA, colB, colC = st.columns(3)
    with colA:
        st.session_state["strict_lossless"] = st.checkbox("Strict 무손실(coverage=1.0)", value=st.session_state["strict_lossless"])
    with colB:
        st.session_state["split_long_articles"] = st.checkbox("롱 조문 분할(문단 경계)", value=st.session_state["split_long_articles"])
    with colC:
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

        # 1) detect → 2) parse
        doc_type = detect_doc_type(text)
        result: ParseResult = parse_document(text, doc_type=doc_type)

        # 3) tree-repair (부모-자식 경계 수복)
        issues_before = validate_tree(result)
        rep_diag = repair_tree(result)
        issues_after = validate_tree(result)

        # 4) chunks (최하위 노드 = article만 쓸 거지만, 파이프라인은 그대로)
        law_name = guess_law_name(text)
        chunks, mk_diag = make_chunks(
            result=result,
            mode="article_only",
            source_file=source_file,
            law_name=law_name,
            include_front_matter=st.session_state["include_front_matter"],
            include_headnotes=st.session_state["include_headnotes"],
            include_gap_fallback=st.session_state["include_gap_fallback"],
            allowed_headnote_levels=list(st.session_state["allowed_headnote_levels"]),
            min_headnote_len=int(st.session_state["min_headnote_len"]),
            min_gap_len=int(st.session_state["min_gap_len"]),
            strict_lossless=bool(st.session_state["strict_lossless"]),
            split_long_articles=bool(st.session_state["split_long_articles"]),
            split_threshold_chars=int(st.session_state["split_threshold_chars"]),
            tail_merge_min_chars=int(st.session_state["tail_merge_min_chars"]),
        )

        st.session_state.update({
            "doc_type": doc_type,
            "law_name": law_name,
            "result": result,
            "chunks": chunks,
            "issues": issues_after,
            "parsed": True,
            "debug": {
                "tree_repair": {
                    "issues_before": len(issues_before),
                    "issues_after": len(issues_after),
                    **rep_diag,
                },
                "make_chunks_diag": mk_diag or {},
            }
        })

    # ───────── 결과 표시 + 다운로드 ───────── #
    if st.session_state["parsed"]:
        cov = _coverage(st.session_state["chunks"], len(st.session_state["result"].full_text))
        arts = [c for c in st.session_state["chunks"] if c.meta.get("type") == "article"]

        with st.container():
            st.markdown("<div class='toolbar'>", unsafe_allow_html=True)
            c1,c2,_ = st.columns([5,1.2,3])
            with c1:
                st.write(
                    f"**파일:** {st.session_state['source_file']}  |  "
                    f"**doc_type:** {st.session_state['doc_type']}  |  "
                    f"**law_name:** {st.session_state['law_name'] or 'N/A'}  |  "
                    f"**chunks:** {len(st.session_state['chunks'])} (article {len(arts)})  |  "
                    f"**coverage:** {cov:.6f}"
                )
            with c2:
                jsonl_bytes = wr.to_jsonl(st.session_state["chunks"]).encode("utf-8")
                st.download_button("JSONL 다운로드", data=jsonl_bytes,
                                   file_name=f"{st.session_state['source_file'].rsplit('.',1)[0]}_chunks.jsonl",
                                   mime="application/json", key="dl-jsonl-top")
            st.markdown("</div>", unsafe_allow_html=True)

        # 원문 전체 + 최하위 노드만 중괄호 감싸기
        html = render_brackets_only(
            text=st.session_state["result"].full_text,
            chunks=st.session_state["chunks"],
        )
        st.markdown('<div class="docwrap">' + html + "</div>", unsafe_allow_html=True)

        # 경고 노출(있다면)
        if st.session_state["issues"]:
            st.caption(f"검증 경고: {len(st.session_state['issues'])}건 (파서 레벨 경계/레벨링 이슈)")

    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
