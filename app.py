# -*- coding: utf-8 -*-
import io
import json
import time
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st

# 패키지 레이아웃(권장)
from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import (
    validate_tree,
    make_chunks,
    guess_law_name,
    repair_tree,  # 트리 수복
)
from parser_core.schema import ParseResult, Node, Chunk
from exporters import writers as wr

APP_TITLE = "Thai Legal Preprocessor — RAG-ready (lossless + tree-repair + JSON Explorer)"

# ───────────────────────────── UI helpers ───────────────────────────── #

def _inject_css():
    st.markdown(
        """
        <style>
          .block-container { max-width: 1400px !important; padding-top: .75rem; }
          .toolbar { position: sticky; top: 0; z-index: 10; padding: .5rem 0 .75rem;
                     background: var(--background-color); border-bottom: 1px solid rgba(128,128,128,.25); }
          .code-like { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: pre-wrap; }
          .json-pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space: pre; font-size: 12px; }
          .muted { color: #6b7280; }
          .kpi { display:inline-block; padding:2px 8px; border-radius:12px; background:#f3f4f6; margin-right:6px; }
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
        # 손실 방지 / 노이즈 억제 옵션 (REPORT에 기록)
        "include_front_matter": True,
        "include_headnotes": True,
        "include_gap_fallback": True,
        "allowed_headnote_levels": ["ภาค","ลักษณะ","หมวด","ส่วน","บท"],
        "min_headnote_len": 24,
        "min_gap_len": 24,
        # Strict 무손실(coverage=1.0 보장 의도)
        "strict_lossless": True,
        # 롱 조문 보조분할 + tail 병합
        "split_long_articles": True,     # ON
        "split_threshold_chars": 1500,
        "tail_merge_min_chars": 200,
        # 탐색기 옵션
        "filter_query": "",
        "show_text_preview_chars": 260,
        "expand_first_n_groups": 6,
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

def _group_chunks(chunks: List[Chunk]) -> Dict[str, List[Chunk]]:
    groups: Dict[str, List[Chunk]] = {}
    for c in chunks:
        key = c.meta.get("section_uid") or c.meta.get("section_label") or c.meta.get("type", "unknown")
        groups.setdefault(key, []).append(c)
    for k in groups:
        groups[k].sort(key=lambda x: (x.span_start, x.span_end))
    return dict(sorted(groups.items(), key=lambda kv: (kv[1][0].span_start if kv[1] else 10**12)))

def _chunk_to_json(c: Chunk) -> Dict[str, Any]:
    return {
        "text": c.text,
        "span": [c.span_start, c.span_end],
        "breadcrumbs": c.breadcrumbs,
        "meta": c.meta,
        "node_ids": c.node_ids,
    }

def _render_json_explorer(chunks: List[Chunk]):
    st.subheader("결과 탐색기 (JSON 노드별 접기/펼치기)")

    qcol1, qcol2, qcol3 = st.columns([2,1,1])
    with qcol1:
        st.session_state["filter_query"] = st.text_input("필터(섹션/메타/텍스트에 포함되는 키워드)", value=st.session_state["filter_query"])
    with qcol2:
        st.session_state["show_text_preview_chars"] = st.number_input("텍스트 미리보기 길이", min_value=80, max_value=2000, value=st.session_state["show_text_preview_chars"], step=20)
    with qcol3:
        st.session_state["expand_first_n_groups"] = st.number_input("처음 펼칠 그룹 수", min_value=0, max_value=50, value=st.session_state["expand_first_n_groups"], step=1)

    groups = _group_chunks(chunks)
    q = (st.session_state["filter_query"] or "").strip()

    for gi, (gkey, items) in enumerate(groups.items(), 1):
        # 그룹 헤더 요약
        types = {}
        for x in items:
            t = x.meta.get("type", "article")
            types[t] = types.get(t, 0) + 1
        type_line = ", ".join(f"{t}:{cnt}" for t, cnt in types.items())

        # 필터
        def _match_group():
            if not q:
                return True
            txt = f"{gkey} {type_line} " + " ".join(x.meta.get("section_label","") for x in items)
            if q.lower() in txt.lower():
                return True
            for x in items[:3]:
                if q.lower() in x.text.lower():
                    return True
            return False

        if not _match_group():
            continue

        expand = gi <= int(st.session_state["expand_first_n_groups"])
        with st.expander(f"{gkey}  ·  {type_line}", expanded=expand):
            for ci, c in enumerate(items, 1):
                meta = c.meta.copy()
                preview = (c.text[:int(st.session_state["show_text_preview_chars"])] + ("…" if len(c.text) > int(st.session_state["show_text_preview_chars"]) else ""))
                with st.expander(f"#{ci:02d} {meta.get('section_label','')}  ·  {meta.get('type','article')}  ·  span={c.span_start}:{c.span_end}", expanded=False):
                    cols = st.columns([3,1])
                    with cols[0]:
                        st.markdown(
                            f"<div class='kpi'>series_index: {meta.get('series_index','1')}</div>"
                            f"<div class='kpi'>series_total: {meta.get('series_total', meta.get('part','1'))}</div>"
                            f"<div class='kpi'>retrieval_weight: {meta.get('retrieval_weight','')}</div>"
                            f"<div class='kpi muted'>doc_type: {meta.get('doc_type','')}</div>",
                            unsafe_allow_html=True
                        )
                        st.json(_chunk_to_json(c))
                    with cols[1]:
                        st.caption("텍스트 미리보기")
                        st.markdown(f"<div class='code-like'>{preview}</div>", unsafe_allow_html=True)

def main():
    _inject_css()
    _ensure_state()

    st.title(APP_TITLE)
    st.caption("Upload UTF-8 Thai legal .txt → [파싱] → lossless chunks + tree-repair → JSON Explorer (node-wise)")

    uploaded = st.file_uploader("텍스트 파일 업로드 (.txt, UTF-8)", type=["txt"])

    # 옵션
    colA, colB, colC = st.columns(3)
    with colA:
        st.session_state["mode"] = st.selectbox("청크 모드", options=["article_only"], index=0)
    with colB:
        st.session_state["strict_lossless"] = st.checkbox("Strict 무손실(coverage=1.0)", value=st.session_state["strict_lossless"])
    with colC:
        st.session_state["include_headnotes"] = st.checkbox("headnote 포함", value=st.session_state["include_headnotes"])

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.session_state["include_front_matter"] = st.checkbox("front matter 포함", value=st.session_state["include_front_matter"])
    with oc2:
        st.session_state["include_gap_fallback"] = st.checkbox("gap-sweeper 포함", value=st.session_state["include_gap_fallback"])
    with oc3:
        st.session_state["allowed_headnote_levels"] = st.multiselect(
            "headnote 허용 레벨", options=["ภาค","ลักษณะ","หมวด","ส่วน","บท"], default=st.session_state["allowed_headnote_levels"]
        )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.session_state["min_headnote_len"] = st.number_input("headnote 최소 길이(문자)", min_value=0, max_value=400, value=st.session_state["min_headnote_len"])
    with sc2:
        st.session_state["min_gap_len"] = st.number_input("gap 보강 최소 길이(문자)", min_value=0, max_value=400, value=st.session_state["min_gap_len"])
    with sc3:
        st.session_state["split_long_articles"] = st.checkbox("롱 조문 보조분할(문단 경계)", value=st.session_state["split_long_articles"])

    sc4, sc5 = st.columns(2)
    with sc4:
        st.session_state["split_threshold_chars"] = st.number_input("보조분할 임계값(문자)", min_value=600, max_value=6000, value=st.session_state["split_threshold_chars"], step=100)
    with sc5:
        st.session_state["tail_merge_min_chars"] = st.number_input("tail 병합 최소 길이(문자)", min_value=0, max_value=600, value=st.session_state["tail_merge_min_chars"], step=10)

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

        # 3) (A단계) tree-repair 이전/이후 이슈 계측
        issues_before = validate_tree(result)
        rep_diag = repair_tree(result)  # 부모 span 확장/축소 + 경계 정규화 (무손실)
        issues_after = validate_tree(result)
        t4 = time.time()

        # 4) chunks (+ diag)
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
            "strict_lossless": bool(st.session_state["strict_lossless"]),
            "split_long_articles": bool(st.session_state["split_long_articles"]),
            "split_threshold_chars": int(st.session_state["split_threshold_chars"]),
            "tail_merge_min_chars": int(st.session_state["tail_merge_min_chars"]),
        }
        st.session_state["debug"] = {
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

    # toolbar / export + JSON Explorer
    if st.session_state["parsed"]:
        cov = _coverage(st.session_state["chunks"], len(st.session_state["result"].full_text))
        with st.container():
            st.markdown("<div class='toolbar'>", unsafe_allow_html=True)
            c1,c2,c3,_ = st.columns([4,1.2,1.2,2])
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

        _render_json_explorer(st.session_state["chunks"])

        if st.session_state["issues"]:
            st.caption(f"검증 경고: {len(st.session_state['issues'])}건 (REPORT.json에 상세 기록됨)")
    else:
        st.info("파일을 올린 뒤 **[파싱]** 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
