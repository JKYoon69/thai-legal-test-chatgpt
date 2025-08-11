# -*- coding: utf-8 -*-
import io
import json
import html
from types import SimpleNamespace
from typing import List, Dict, Any, Tuple

import streamlit as st

from parser_core.postprocess import (
    make_chunks, validate_tree, repair_tree, merge_small_trailing_parts, guess_law_name, Chunk
)

APP_TITLE = "Thai Legal Preprocessor — RAG-ready (lossless + debug)"


# ------------------------------ UI Helpers ------------------------------

def _human_int(n: int) -> str:
    return f"{n:,}"


def _guess_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("utf-8", errors="ignore")


def _naive_summary(text: str, maxlen: int = 80) -> str:
    t = " ".join(text.strip().split())
    if not t:
        return ""
    # 문장 끝(태국어 포함)을 대충 감지
    for sep in ["。", "!", "?", "ฯ", ".", "；", ";"]:
        if sep in t:
            t = t.split(sep)[0]
            break
    if len(t) > maxlen:
        t = t[:maxlen - 1] + "…"
    return t


def chunk_summary_text(ch: Chunk, maxlen: int = 80) -> str:
    meta = ch.meta or {}
    for k in ("desc", "summary", "llm_summary", "section_label"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            v2 = " ".join(v.strip().split())
            return v2 if len(v2) <= maxlen else (v2[:maxlen - 1] + "…")
    return _naive_summary(ch.text, maxlen=maxlen)


# ------------------------------ Annotated Renderer ------------------------------

def render_annotated(text: str, chunks: List[Chunk]) -> str:
    """
    원문에 braces(경계/오버랩)와 'chunk #### — summary'를 삽입하여 HTML 문자열로 반환
    - 초록(연두) braces: 청크 경계
    - 핑크 braces: 오버랩 영역 (span 포함 구간 중 core_span 바깥)
    """
    # 색상
    GREEN = "#9EE37D"      # 연두
    PINK = "#FF7EB6"       # 핑크
    NUM_COLOR = GREEN

    # 이벤트 테이블: pos -> [(order, html)]
    events: Dict[int, List[Tuple[int, str]]] = {}

    def add_event(pos: int, order: int, html_frag: str):
        events.setdefault(pos, []).append((order, html_frag))

    # 전체 경계 brace + 오버랩 brace + 닫는 쪽에 번호/요약
    for idx, ch in enumerate(chunks, start=1):
        s = int(ch.span_start)
        e = int(ch.span_end)
        core_s, core_e = ch.meta.get("core_span", [s, e])
        ol = int(ch.meta.get("overlap_left", 0) or 0)
        or_ = int(ch.meta.get("overlap_right", 0) or 0)

        # 1) 청크 전체 경계 (연두)
        add_event(s, 10, f'<span style="color:{GREEN}; font-weight:700">{{</span>')
        add_event(e, 90, f'</span><span style="color:{GREEN}; font-weight:700">}}</span>')

        # 2) 오버랩 (핑크) - 왼쪽/오른쪽 각각 브래킷으로 감싸 표기
        if ol > 0 and s < core_s:
            # 왼쪽 오버랩: [s, core_s)
            add_event(s, 20, f'<span style="color:{PINK}; font-weight:700">{{</span>')
            add_event(core_s, 21, f'<span style="color:{PINK}; font-weight:700">}}</span>')

        if or_ > 0 and core_e < e:
            # 오른쪽 오버랩: [core_e, e)
            add_event(core_e, 80, f'<span style="color:{PINK}; font-weight:700">{{</span>')
            add_event(e, 81, f'<span style="color:{PINK}; font-weight:700">}}</span>')

        # 3) 닫는 brace 뒤에 " chunk #### — summary"
        summary = html.escape(chunk_summary_text(ch, maxlen=80))
        num = f"{idx:04d}"
        tail = f'<span style="color:{NUM_COLOR}; font-weight:700">  chunk {num} — {summary}</span>'
        add_event(e, 95, tail)

    # 조립
    safe = html.escape(text)
    out = []
    last = 0
    # events는 원문 기준 위치 인덱스, safe와 원문 길이는 동일(escape 전 분할 금지)
    # => safe에서 last:pos 구간 잘라 붙이고, 이벤트 HTML을 이어붙인다.
    for pos in sorted(events.keys()):
        if pos > len(safe):
            pos = len(safe)
        out.append(safe[last:pos])
        snippets = sorted(events[pos], key=lambda x: x[0])
        out.extend([frag for _, frag in snippets])
        last = pos
    out.append(safe[last:])

    styles = """
    <style>
      .mono { white-space: pre-wrap; font-family: ui-monospace, Menlo, Consolas, "Noto Sans Thai", "Noto Serif Thai", "Apple SD Gothic Neo", monospace; line-height: 1.6; }
    </style>
    """
    return styles + '<div class="mono">' + "".join(out) + "</div>"


# ------------------------------ Downloads ------------------------------

def to_jsonl_compat(chunks: List[Chunk]) -> bytes:
    buf = io.StringIO()
    for ch in chunks:
        row = {
            "text": ch.text,
            "span_start": ch.span_start,
            "span_end": ch.span_end,
            "meta": ch.meta,
            "breadcrumbs": ch.breadcrumbs,
        }
        buf.write(json.dumps(row, ensure_ascii=False) + "\n")
    return buf.getvalue().encode("utf-8")


def to_json_report(report: Dict[str, Any]) -> bytes:
    return (json.dumps(report, ensure_ascii=False, indent=2)).encode("utf-8")


# ------------------------------ Main ------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    up = st.file_uploader("Upload UTF-8 Thai legal text (.txt)", type=["txt"])
    if not up:
        st.info("텍스트 파일을 업로드하세요.")
        return

    text = _guess_text(up.read())
    law_name = guess_law_name(text)

    with st.sidebar:
        st.subheader("옵션")
        split_threshold = st.number_input("분할 임계값(문자)", min_value=600, max_value=4000, value=1500, step=50)
        overlap_chars = st.number_input("오버랩 길이", min_value=0, max_value=400, value=200, step=10)
        tail_merge_min = st.number_input("tail 병합 최소 길이(문자)", min_value=0, max_value=400, value=200, step=10)
        skip_short = st.number_input("짧은 기사 흡수 임계(문자)", min_value=0, max_value=400, value=180, step=10)
        pack_target = st.number_input("패킹 target 최소 길이", min_value=0, max_value=1000, value=450, step=10)
        pack_max_members = st.number_input("패킹 최대 멤버 수", min_value=2, max_value=6, value=3, step=1)

        st.markdown("---")
        strict = st.checkbox("Strict 무손실(coverage=1.0 근접)", value=True)
        include_head = st.checkbox("Headnote(ส่วน/หมวด 등) 표시", value=True)

    # ParseResult 모사(텍스트 기반)
    result = SimpleNamespace(full_text=text, doc_type="code")

    with st.spinner("파싱/청킹 중…"):
        chunks, diag = make_chunks(
            result=result,
            mode="article_only",
            source_file=up.name,
            law_name=law_name,
            include_front_matter=True,
            include_headnotes=include_head,
            include_gap_fallback=True,
            strict_lossless=strict,
            split_long_articles=True,
            split_threshold_chars=int(split_threshold),
            overlap_chars=int(overlap_chars),
            tail_merge_min_chars=int(tail_merge_min),
            skip_short_chars=int(skip_short),
            pack_target_min=int(pack_target),
            pack_max_members=int(pack_max_members),
            soft_cut=True,
        )

    st.success(f"완료! | 파일: **{up.name}** | doc_type: {getattr(result,'doc_type','')} | law_name: {law_name or 'N/A'} | chunks: {_human_int(len(chunks))}")

    # 리포트 구성
    report = {
        "file": up.name,
        "doc_type": getattr(result, "doc_type", ""),
        "law_name": law_name,
        "chunks": len(chunks),
        "coverage": round(1.0, 6) if strict else None,
        "stats": diag,
    }

    # 다운로드
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("JSONL 다운로드", data=to_jsonl_compat(chunks), file_name=f"{up.name.rsplit('.',1)[0]}_chunks.jsonl", mime="application/jsonl")
    with c2:
        st.download_button("REPORT.json 다운로드", data=to_json_report(report), file_name=f"{up.name.rsplit('.',1)[0]}_REPORT.json", mime="application/json")

    # 본문 시각화
    st.markdown("### 본문 미리보기 (청크 경계/오버랩 + 번호/요약)")
    html_str = render_annotated(text, chunks)
    st.markdown(html_str, unsafe_allow_html=True)

    # 디버그 패널
    with st.expander("디버그(진단) 보기"):
        st.json(report)


if __name__ == "__main__":
    main()
