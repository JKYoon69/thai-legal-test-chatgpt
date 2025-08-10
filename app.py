# -*- coding: utf-8 -*-
import time
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st

# 파이프라인 모듈
from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import (
    validate_tree,
    make_chunks,
    guess_law_name,
    repair_tree,  # 트리 수복
)
from parser_core.schema import ParseResult, Node, Chunk
from exporters.writers import to_jsonl, make_debug_report  # REPORT.json 생성

APP_TITLE = "Thai Legal Preprocessor — Full Text + Brackets (leaf/chunk)"

# ───────────────────────────── UI helpers ───────────────────────────── #

def _inject_css():
    st.markdown(
        """
        <style>
          .block-container { max-width: 1400px !important; padding-top: .75rem; }
          .docwrap { border: 1px solid rgba(107,114,128,.25); border-radius: 10px; padding: 16px; }
          .doc { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                 font-size: 13.5px; line-height: 1.6; position: relative; }

          /* 연두색 중괄호(스팬 전체 감쌈) */
          .bracket-block { position: relative; padding: 0 12px; border-radius: 6px; margin: 0 1px; }
          .bracket-block::before, .bracket-block::after {
            content: "{"; position: absolute; top: -1px; bottom: -1px; width: 10px;
            color: #a3e635; opacity: .90; font-weight: 800;
          }
          .bracket-block::before { left: 0; }
          .bracket-block::after  { right: 0; transform: scaleX(-1); }

          /* 닫는 표시 + 번호 */
          .close-tail { color: #a3e635; font-weight: 800; }
          .dlbar { display:flex; gap:10px; align-items:center; margin: .6rem 0 .6rem; }
          .muted { color:#6b7280; font-size: 12px; }

          /* 파싱 버튼 크게/선명하게 */
          .parse-line { margin-top: .25rem; margin-bottom: .5rem; }
          .parse-line button { background: #22c55e !important; color: white !important; border: 0 !important;
                               padding: .75rem 1.2rem !important; font-weight: 800 !important; font-size: 16px !important;
                               border-radius: 10px !important; width: 100%; }
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
        # 파이프라인 옵션
        "strict_lossless": True,
        "split_long_articles": True,
        "split_threshold_chars": 1500,
        "tail_merge_min_chars": 200,
        "overlap_chars": 200,         # 파트 사이 오버랩
        # 추출 범위
        "include_front_matter": True,
        "include_headnotes": True,
        "include_gap_fallback": True,
        "min_headnote_len": 24,
        "min_gap_len": 24,
        "allowed_headnote_levels": ["ภาค","ลักษณะ","หมวด","ส่วน","บท"],
        # 표시 옵션
        "bracket_mode": "리프(มาตรา/ข้อ)",     # 또는 "청크"
        "number_scope": "article만",          # 또는 "전체 청크"
        "bracket_front_matter": True,         # 문서 첫부분도 중괄호로 볼지
        # 내부 저장(다운로드용)
        "report_json_str": "",
        "chunks_jsonl_str": "",
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

# ───────────────────────────── Bracket renderers ───────────────────────────── #

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

def render_leaf_brackets(text: str, result: ParseResult, *, add_newline=True, add_number=False) -> str:
    """
    원문 전체를 그대로 출력하되,
    제일 하위 노드(มาตรา/ข้อ)의 스팬을 연두 중괄호로 감싼다.
    """
    N = len(text)
    leaves = _collect_article_leaves(result.root)

    parts: List[str] = []
    cur = 0
    leaf_idx = 0
    for lf in leaves:
        s, e = int(lf.span_start), int(lf.span_end)
        if s < 0 or e > N or e <= s:
            continue
        if s > cur:
            parts.append(_html_escape(text[cur:s]))
        seg = _html_escape(text[s:e])
        label = f"{lf.label or ''}{(' ' + lf.num) if lf.num else ''}".strip()
        # breadcrumbs
        crumbs = []
        p = result.node_map.get(lf.parent_id)
        while p and p.label and p.label != "root":
            crumbs.append(f"{p.label}{(' ' + p.num) if p.num else ''}".strip())
            p = result.node_map.get(p.parent_id)
        crumbs = " / ".join(reversed(crumbs))
        title = label if not crumbs else f"{label} — {crumbs}"

        parts.append(f'<span class="bracket-block" title="{_html_escape(title)}">{seg}</span>')
        # 닫는 괄호 뒤 표시
        tail = ""
        if add_number:
            leaf_idx += 1
            tail = f' <span class="close-tail">}} leaf {leaf_idx:04d}</span>'
        else:
            tail = ' <span class="close-tail">}}</span>'
        if add_newline:
            tail += "<br/>"
        parts.append(tail)
        cur = e
    if cur < N:
        parts.append(_html_escape(text[cur:N]))
    return '<div class="doc">' + "".join(parts) + "</div>"

def render_chunk_brackets(
    text: str,
    chunks: List[Chunk],
    *,
    number_scope: str = "article만",   # 또는 "전체 청크"
    bracket_front_matter: bool = True,
    add_newline: bool = True,
) -> str:
    """
    원문 전체를 그대로 출력하되,
    '청크' 스팬을 연두 중괄호로 감싸고 } 뒤에 'chunk 0001'을 붙인다.
    - number_scope='article만'이면 article 타입만 번호를 매김(표시는 article만)
    - number_scope='전체 청크'이면 front_matter/headnote도 표시+번호 매김
    """
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
        tail = f' <span class="close-tail">}} chunk {idx:04d}</span>'
        if add_newline:
            tail += "<br/>"
        parts.append(tail)
        cur = e
    if cur < N:
        parts.append(_html_escape(text[cur:N]))
    return '<div class="doc">' + "".join(parts) + "</div>"

# ───────────────────────────── Main ───────────────────────────── #

def main():
    _inject_css()
    _ensure_state()

    st.title(APP_TITLE)
    st.caption("Upload UTF-8 Thai legal .txt → [파싱] → 원문 전체 + 브래킷 표시(리프/청크)")

    uploaded = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])

    # ▶ 파싱 버튼: 파일명 바로 아래, 크고 눈에 띄게
    with st.container():
        st.markdown('<div class="parse-line">', unsafe_allow_html=True)
        parse_clicked = st.button("파싱", key="parse_btn_top")
        st.markdown('</div>', unsafe_allow_html=True)

    # 옵션: 한 줄에 3개(분할 임계값, tail 병합, 오버랩)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state["split_threshold_chars"] = st.number_input("분할 임계값(문자)", min_value=600, max_value=6000, value=st.session_state["split_threshold_chars"], step=100)
    with col2:
        st.session_state["tail_merge_min_chars"] = st.number_input("tail 병합 최소 길이(문자)", min_value=0, max_value=600, value=st.session_state["tail_merge_min_chars"], step=10)
    with col3:
        st.session_state["overlap_chars"] = st.number_input("오버랩(문맥) 길이", min_value=0, max_value=800, value=st.session_state["overlap_chars"], step=25)

    # 표시 옵션
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.session_state["bracket_mode"] = st.selectbox("브래킷 기준", ["리프(มาตรา/ข้อ)", "청크"], index=0)
    with oc2:
        st.session_state["number_scope"] = st.selectbox("번호 기준", ["article만", "전체 청크"], index=0)
    with oc3:
        st.session_state["bracket_front_matter"] = st.checkbox("문서 첫부분도 중괄호(Front matter)", value=st.session_state["bracket_front_matter"])

    # 나머지 옵션(간단)
    oc4, oc5, _ = st.columns(3)
    with oc4:
        st.session_state["strict_lossless"] = st.checkbox("Strict 무손실(coverage=1.0)", value=st.session_state["strict_lossless"])
    with oc5:
        st.session_state["split_long_articles"] = st.checkbox("롱 조문 분할(문단 경계)", value=st.session_state["split_long_articles"])

    # 파일 적재
    if uploaded is not None:
        try:
            st.session_state["text"] = uploaded.read().decode("utf-8")
            st.session_state["source_file"] = uploaded.name
        except UnicodeDecodeError:
            st.error("파일 인코딩을 UTF-8로 저장해주세요.")
            return

    # 파싱 실행
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

        # 3) tree-repair
        issues_before = validate_tree(result)
        rep_diag = repair_tree(result)
        issues_after = validate_tree(result)

        # 4) chunks (오버랩 + 소프트컷 적용)
        law_name = guess_law_name(text)
        chunks, mk_diag = make_chunks(
            result=result,
            mode="article_only",
            source_file=source_file,
            law_name=law_name,
            include_front_matter=True,
            include_headnotes=True,
            include_gap_fallback=True,
            allowed_headnote_levels=list(st.session_state["allowed_headnote_levels"]),
            min_headnote_len=int(st.session_state["min_headnote_len"]),
            min_gap_len=int(st.session_state["min_gap_len"]),
            strict_lossless=bool(st.session_state["strict_lossless"]),
            split_long_articles=bool(st.session_state["split_long_articles"]),
            split_threshold_chars=int(st.session_state["split_threshold_chars"]),
            tail_merge_min_chars=int(st.session_state["tail_merge_min_chars"]),
            overlap_chars=int(st.session_state["overlap_chars"]),
            soft_cut=True,
        )

        # 커버리지 계산 + 다운로드용 파일 생성
        cov = _coverage(chunks, len(result.full_text))
        report_str = make_debug_report(
            parse_result=result,
            chunks=chunks,
            source_file=source_file,
            law_name=law_name or "",
            run_config={
                "strict_lossless": bool(st.session_state["strict_lossless"]),
                "split_long_articles": bool(st.session_state["split_long_articles"]),
                "split_threshold_chars": int(st.session_state["split_threshold_chars"]),
                "tail_merge_min_chars": int(st.session_state["tail_merge_min_chars"]),
                "overlap_chars": int(st.session_state["overlap_chars"]),
                "bracket_mode": st.session_state["bracket_mode"],
                "number_scope": st.session_state["number_scope"],
                "bracket_front_matter": bool(st.session_state["bracket_front_matter"]),
            },
            debug={
                "tree_repair": {
                    "issues_before": len(issues_before),
                    "issues_after": len(issues_after),
                    **rep_diag,
                },
                "make_chunks_diag": mk_diag or {},
                "coverage_calc": {"coverage": cov},
            },
        )
        chunks_str = to_jsonl(chunks)

        # 상태 저장
        st.session_state.update({
            "doc_type": doc_type,
            "law_name": law_name,
            "result": result,
            "chunks": chunks,
            "issues": issues_after,
            "parsed": True,
            "report_json_str": report_str,
            "chunks_jsonl_str": chunks_str,
        })

    # ───────── 결과 표시 ───────── #
    if st.session_state["parsed"]:
        cov = _coverage(st.session_state["chunks"], len(st.session_state["result"].full_text))
        arts = [c for c in st.session_state["chunks"] if c.meta.get("type") == "article"]

        # 간단 정보
        st.write(
            f"**파일:** {st.session_state['source_file']}  |  "
            f"**doc_type:** {st.session_state['doc_type']}  |  "
            f"**law_name:** {st.session_state['law_name'] or 'N/A'}  |  "
            f"**chunks:** {len(st.session_state['chunks'])} (article {len(arts)})  |  "
            f"**coverage:** {cov:.6f}"
        )

        # 다운로드 바 (한 줄)
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

        # 원문 전체 + 브래킷 표시
        if st.session_state["bracket_mode"] == "청크":
            html = render_chunk_brackets(
                text=st.session_state["result"].full_text,
                chunks=st.session_state["chunks"],
                number_scope=st.session_state["number_scope"],
                bracket_front_matter=bool(st.session_state["bracket_front_matter"]),
                add_newline=True,
            )
        else:
            html = render_leaf_brackets(
                text=st.session_state["result"].full_text,
                result=st.session_state["result"],
                add_newline=True,
                add_number=False,  # 리프 모드에선 번호 기본 비활성(원하면 True로)
            )

        st.markdown('<div class="docwrap">' + html + "</div>", unsafe_allow_html=True)

        # 경고 노출(있다면)
        if st.session_state["issues"]:
            st.caption(f"검증 경고: {len(st.session_state['issues'])}건 (파서 레벨 경계/레벨링 이슈)")

    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
