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
from parser_core.schema import ParseResult, Chunk
from exporters import writers as wr


APP_TITLE = "Thai Legal Preprocessor — Full Document View (article-level highlights)"

# ───────────────────────────── UI helpers ───────────────────────────── #

def _inject_css():
    st.markdown(
        """
        <style>
          .block-container { max-width: 1400px !important; padding-top: .75rem; }
          .toolbar { position: sticky; top: 0; z-index: 10; padding: .5rem 0 .75rem;
                     background: var(--background-color); border-bottom: 1px solid rgba(128,128,128,.25); }
          .docwrap { border: 1px solid rgba(107,114,128,.25); border-radius: 10px; padding: 16px; }
          .doc { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 13.5px; line-height: 1.6; position: relative; }
          /* 교차 배경 */
          .article-block { position: relative; border-radius: 6px; padding: 0 2px; }
          .article-even { background: rgba(34,197,94,0.11); }
          .article-odd  { background: rgba(59,130,246,0.10); }
          /* 녹색 중괄호 스타일 */
          .bracket-block { position: relative; padding: 0 10px; }
          .bracket-block::before, .bracket-block::after {
            content: "{"; position: absolute; top: -1px; bottom: -1px; width: 8px;
            color: #16a34a; font-weight: 700; opacity: .6;
          }
          .bracket-block::before { left: 0; }
          .bracket-block::after  { right: 0; transform: scaleX(-1); }
          /* orphan gap 표시 (붉은 점선 배경) */
          .gap-block {
            background-image: repeating-linear-gradient( -45deg, rgba(239,68,68,0.16) 0, rgba(239,68,68,0.16) 6px, transparent 6px, transparent 12px );
            border-bottom: 1px dotted rgba(239,68,68,.8);
          }
          .kpi { display:inline-block; padding:2px 8px; border-radius:12px; background:#f3f4f6; margin-right:6px; }
          .muted { color:#6b7280; }
          .pill { display:inline-block; padding:0 6px; line-height:18px; height:18px; font-size:12px; border-radius:999px; background:#eef2ff; color:#3730a3; margin-left:6px; }
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
        "mode": "article_only",
        "include_front_matter": True,
        "include_headnotes": True,
        "include_gap_fallback": True,
        "allowed_headnote_levels": ["ภาค","ลักษณะ","หมวด","ส่วน","บท"],
        "min_headnote_len": 24,
        "min_gap_len": 24,
        "strict_lossless": True,
        "split_long_articles": True,
        "split_threshold_chars": 1500,
        "tail_merge_min_chars": 200,
        # 뷰 옵션
        "viz_style": "교차 배경색",  # or "녹색 중괄호"
        "shade_headnotes": False,    # headnote 흐리게 표기할지
        "show_gaps": True,           # orphan_gap 표시
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

# ───────────────────────────── Full doc renderer ───────────────────────────── #

def _render_full_document(text: str, chunks: List[Chunk], *, viz_style: str = "교차 배경색",
                          show_gaps: bool = True, shade_headnotes: bool = False) -> str:
    """
    원문 전체를 한 번에 렌더.
    - 제일 하위 노드(조문/ข้อ)만 색 블록 or 중괄호로 표시
    - orphan_gap은 붉은 점선 배경
    - headnote는 옵션(옅게)
    """
    N = len(text)
    # 관심 구간 수집
    articles = [c for c in chunks if c.meta.get("type") == "article"]
    gaps     = [c for c in chunks if c.meta.get("type") == "orphan_gap"]
    headnts  = [c for c in chunks if c.meta.get("type") == "headnote"] if shade_headnotes else []

    # 이벤트(경계) 구성
    Event = Tuple[int, str, str, Optional[Chunk]]  # (pos, 'start'|'end', kind, chunk)
    evts: List[Event] = []
    for c in articles:
        evts.append((c.span_start, "start", "article", c))
        evts.append((c.span_end,   "end",   "article", c))
    if show_gaps:
        for c in gaps:
            evts.append((c.span_start, "start", "gap", c))
            evts.append((c.span_end,   "end",   "gap", c))
    for c in headnts:
        evts.append((c.span_start, "start", "head", c))
        evts.append((c.span_end,   "end",   "head", c))

    # 포지션/우선순위 정렬: 같은 위치면 end 먼저 처리
    priority = {"end": 0, "start": 1}
    evts.sort(key=lambda x: (x[0], priority[x[1]]))

    # 스캔
    html_parts: List[str] = []
    pos = 0
    active_article: Optional[Chunk] = None
    active_gap: Optional[Chunk] = None
    active_head: Optional[Chunk] = None
    article_count = 0  # 홀/짝 교차

    def open_tag_for_article(c: Chunk, idx: int) -> str:
        evenodd = "article-even" if (idx % 2 == 0) else "article-odd"
        base_cls = "article-block " + (evenodd if viz_style == "교차 배경색" else "bracket-block")
        meta = c.meta or {}
        tip = []
        if meta.get("section_label"): tip.append(meta["section_label"])
        si, st = meta.get("series_index","1"), meta.get("series_total","1")
        if st and st != "1":
            tip.append(f"(part {si}/{st})")
        if c.breadcrumbs:
            tip.append(" / ".join(c.breadcrumbs))
        title = " — ".join([t for t in tip if t]).strip()
        data_attr = f'data-series-index="{si}" data-series-total="{st}"'
        pill = f'<span class="pill">{si}/{st}</span>' if st and st != "1" else ""
        # 소제목 배지는 시각적 힌트로만, 텍스트는 그대로 보존
        return f'<span class="{base_cls}" title="{_html_escape(title)}" {data_attr}>'  # {pill}는 시각적 배지지만 라인 흐름 의존 → 툴팁만

    def close_tag_for_article() -> str:
        return "</span>"

    def open_tag_for_gap() -> str:
        return '<span class="gap-block" title="orphan_gap (미덮임/strict-fill 영역)">'

    def close_tag_for_gap() -> str:
        return "</span>"

    def open_tag_for_head() -> str:
        # 아주 옅은 회색 배경
        return '<span style="background: rgba(107,114,128,0.08);" title="headnote">'

    def close_tag_for_head() -> str:
        return "</span>"

    for (p, typ, kind, ref) in evts:
        if p > N: p = N
        if p > pos:
            seg = _html_escape(text[pos:p])
            # 상태에 따라 감싸기
            if active_gap is not None:
                html_parts.append(open_tag_for_gap() + seg + close_tag_for_gap())
            elif active_article is not None:
                html_parts.append(seg)  # 이미 열린 article span 내부
            elif active_head is not None:
                html_parts.append(open_tag_for_head() + seg + close_tag_for_head())
            else:
                html_parts.append(seg)
            pos = p

        # 상태 갱신: end 먼저
        if typ == "end":
            if kind == "article" and active_article is not None and ref is not None and (ref.span_end == p):
                html_parts.append(close_tag_for_article())
                active_article = None
            elif kind == "gap" and active_gap is not None and ref is not None and (ref.span_end == p):
                # gap은 wrap을 seg마다 열고 닫기 때문에 여기선 무시
                active_gap = None
            elif kind == "head" and active_head is not None and ref is not None and (ref.span_end == p):
                # head도 seg마다 열고 닫기
                active_head = None

        elif typ == "start":
            if kind == "article":
                active_article = ref
                html_parts.append(open_tag_for_article(ref, article_count))
                article_count += 1
            elif kind == "gap":
                active_gap = ref
            elif kind == "head":
                active_head = ref

    # 꼬리
    if pos < N:
        seg = _html_escape(text[pos:N])
        if active_gap is not None:
            html_parts.append(open_tag_for_gap() + seg + close_tag_for_gap())
        elif active_article is not None:
            html_parts.append(seg)
            html_parts.append(close_tag_for_article())
            active_article = None
        elif active_head is not None:
            html_parts.append(open_tag_for_head() + seg + close_tag_for_head())
        else:
            html_parts.append(seg)

    return '<div class="doc">' + "".join(html_parts) + "</div>"

# ───────────────────────────── Main ───────────────────────────── #

def main():
    _inject_css()
    _ensure_state()

    st.title(APP_TITLE)
    st.caption("Upload UTF-8 Thai legal .txt → [파싱] → Full document view with article-level highlights")

    uploaded = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])

    # 옵션 (파이프라인)
    colA, colB, colC = st.columns(3)
    with colA:
        st.session_state["mode"] = st.selectbox("청크 모드", options=["article_only"], index=0)
    with colB:
        st.session_state["strict_lossless"] = st.checkbox("Strict 무손실(coverage=1.0)", value=st.session_state["strict_lossless"])
    with colC:
        st.session_state["split_long_articles"] = st.checkbox("롱 조문 분할(문단 경계)", value=st.session_state["split_long_articles"])

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.session_state["split_threshold_chars"] = st.number_input("분할 임계값(문자)", min_value=600, max_value=6000, value=st.session_state["split_threshold_chars"], step=100)
    with oc2:
        st.session_state["tail_merge_min_chars"] = st.number_input("tail 병합 최소 길이(문자)", min_value=0, max_value=600, value=st.session_state["tail_merge_min_chars"], step=10)
    with oc3:
        st.session_state["include_headnotes"] = st.checkbox("headnote 포함(추출)", value=st.session_state["include_headnotes"])

    oc4, oc5, oc6 = st.columns(3)
    with oc4:
        st.session_state["include_front_matter"] = st.checkbox("front matter 포함(추출)", value=st.session_state["include_front_matter"])
    with oc5:
        st.session_state["include_gap_fallback"] = st.checkbox("gap-sweeper 포함", value=st.session_state["include_gap_fallback"])
    with oc6:
        st.session_state["allowed_headnote_levels"] = st.multiselect(
            "headnote 허용 레벨", options=["ภาค","ลักษณะ","หมวด","ส่วน","บท"], default=st.session_state["allowed_headnote_levels"]
        )

    oc7, oc8, oc9 = st.columns(3)
    with oc7:
        st.session_state["min_headnote_len"] = st.number_input("headnote 최소 길이", min_value=0, max_value=400, value=st.session_state["min_headnote_len"])
    with oc8:
        st.session_state["min_gap_len"] = st.number_input("gap 최소 길이", min_value=0, max_value=400, value=st.session_state["min_gap_len"])
    with oc9:
        st.session_state["viz_style"] = st.selectbox("표시 스타일", options=["교차 배경색", "녹색 중괄호"], index=0)

    oc10, oc11, oc12 = st.columns(3)
    with oc10:
        st.session_state["shade_headnotes"] = st.checkbox("headnote 옅게 표시", value=st.session_state["shade_headnotes"])
    with oc11:
        st.session_state["show_gaps"] = st.checkbox("orphan_gap(미덮임) 표시", value=st.session_state["show_gaps"])
    with oc12:
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

        # 3) tree-repair
        issues_before = validate_tree(result)
        rep_diag = repair_tree(result)
        issues_after = validate_tree(result)
        t4 = time.time()

        # 4) chunks
        law_name = guess_law_name(text)
        chunks, mk_diag = make_chunks(
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
            strict_lossless=bool(st.session_state["strict_lossless"]),
            split_long_articles=bool(st.session_state["split_long_articles"]),
            split_threshold_chars=int(st.session_state["split_threshold_chars"]),
            tail_merge_min_chars=int(st.session_state["tail_merge_min_chars"]),
        )
        t5 = time.time()

        st.session_state.update({
            "doc_type": doc_type,
            "law_name": law_name,
            "result": result,
            "chunks": chunks,
            "issues": issues_after,
            "parsed": True,
            "run_config": {
                "mode": st.session_state["mode"],
                "include_front_matter": st.session_state["include_front_matter"],
                "include_headnotes": st.session_state["include_headnotes"],
                "include_gap_fallback": st.session_state["include_gap_fallback"],
                "allowed_headnote_levels": list(st.session_state["allowed_headnote_levels"]),
                "min_headnote_len": int(st.session_state["min_headnote_len"]),
                "min_gap_len": int(st.session_state["min_gap_len"]),
                "strict_lossless": bool(st.session_state["strict_lossless"]),
                "split_long_articles": bool(st.session_state["split_long_articles"]),
                "split_threshold_chars": int(st.session_state["split_threshold_chars"]),
                "tail_merge_min_chars": int(st.session_state["tail_merge_min_chars"]),
            },
            "debug": {
                "timings_sec": {
                    "detect": round(t2 - t1, 6),
                    "parse": round(t3 - t2, 6),
                    "tree_repair": round(t4 - t3, 6),
                    "make_chunks": round(t5 - t4, 6),
                    "total": round(t5 - t0, 6),
                },
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
        # orphan_gap 통계
        gaps = [c for c in st.session_state["chunks"] if c.meta.get("type") == "orphan_gap"]
        gap_chars = sum((c.span_end - c.span_start) for c in gaps)

        with st.container():
            st.markdown("<div class='toolbar'>", unsafe_allow_html=True)
            c1,c2,c3,_ = st.columns([4,1.2,1.2,2])
            with c1:
                st.write(
                    f"**파일:** {st.session_state['source_file']}  |  "
                    f"**doc_type:** {st.session_state['doc_type']}  |  "
                    f"**law_name:** {st.session_state['law_name'] or 'N/A'}  |  "
                    f"**chunks:** {len(st.session_state['chunks'])}  |  "
                    f"**coverage:** {cov:.6f}  "
                    f"{'(미덮임: ' + str(gap_chars) + '자 / ' + str(len(gaps)) + '구간)' if st.session_state['show_gaps'] and gaps else ''}"
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

        # 원문 전체를 한 번에 렌더
        html = _render_full_document(
            text=st.session_state["result"].full_text,
            chunks=st.session_state["chunks"],
            viz_style=st.session_state["viz_style"],
            show_gaps=bool(st.session_state["show_gaps"]),
            shade_headnotes=bool(st.session_state["shade_headnotes"]),
        )
        st.markdown('<div class="docwrap">' + html + "</div>", unsafe_allow_html=True)

        # 검증 경고
        if st.session_state["issues"]:
            st.caption(f"검증 경고: {len(st.session_state['issues'])}건 (REPORT.json에 상세 기록됨)")
    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
