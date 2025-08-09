import json
from pathlib import Path
import streamlit as st

from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks
from parser_core.schema import ParseResult
from exporters.writers import to_jsonl, make_zip_bundle

st.set_page_config(page_title="Thai Law Parser (Test)", layout="wide")
st.title("📜 Thai Law Parser — 테스트")

with st.sidebar:
    st.markdown("**업로드 → 파싱 → 검토 → 다운로드** 순서로 진행하세요.")
    st.markdown("문서유형 감지 실패 시 수동으로 선택할 수 있어요.")
    st.caption("v0.2 — line-anchored headers, spans-only nodes")

uploaded = st.file_uploader("태국어 법률 문서 업로드 (.txt)", type=["txt"])

if "raw_text" not in st.session_state:
    st.session_state.raw_text = None
    st.session_state.result: ParseResult|None = None

if uploaded:
    raw = uploaded.read().decode("utf-8", errors="ignore")
    st.session_state.raw_text = raw

if st.session_state.raw_text:
    raw_text = st.session_state.raw_text
    with st.expander("원문 미리보기", expanded=False):
        st.text(raw_text[:1500] + ("..." if len(raw_text) > 1500 else ""))

    auto_type = detect_doc_type(raw_text)
    st.write(f"🔎 자동 감지된 문서 유형: **{auto_type}**")

    forced_type = st.selectbox(
        "문서 유형(수정 가능)",
        options=["auto", "code", "act", "royal_decree", "regulation"],
        index=0,
        help="감지가 틀리면 여기서 강제로 지정하세요.",
    )
    if forced_type == "auto":
        forced_type = None

    if st.button("🧩 파싱 실행", use_container_width=True):
        with st.spinner("파싱 중..."):
            result = parse_document(raw_text, forced_type=forced_type)
            st.session_state.result = result

    result: ParseResult|None = st.session_state.result
    if result:
        st.success(
            f"파싱 완료: 노드 {len(result.nodes)}개, 최하위 노드 {result.stats.get('leaf_count', 0)}개"
        )

        # 검증 리포트
        issues = validate_tree(result)
        with st.expander(f"🔍 정합성 검증 (경고 {len(issues)}건)", expanded=False):
            if not issues:
                st.write("문제 없음 ✅")
            else:
                for i, iss in enumerate(issues, 1):
                    st.write(f"{i}. [{iss.level}] {iss.message}")

        # 좌/우 컬럼: 좌(트리), 우(하이라이트)
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.subheader("계층 트리")
            def render_node(node, depth=0):
                pad = "　" * depth
                label = f"{node.level} {node.label}".strip()
                st.write(f"{pad}- **{label}**  (chars {node.span.start}–{node.span.end})")
                for ch in node.children:
                    render_node(ch, depth + 1)

            for n in result.root_nodes:
                render_node(n, 0)

        with right:
            st.subheader("원문 하이라이트")
            leaf_options = [(n.node_id, f"{n.level} {n.label}") for n in result.nodes]
            sel = st.selectbox("노드 선택(해당 부분 하이라이트)", leaf_options, index=0)
            sel_id = sel[0]
            target = next((n for n in result.nodes if n.node_id == sel_id), None)
            if target:
                start = max(0, target.span.start - 200)
                end = min(len(raw_text), target.span.end + 200)
                prefix = raw_text[start:target.span.start]
                body = raw_text[target.span.start:target.span.end]
                suffix = raw_text[target.span.end:end]
                st.markdown(
                    f"…{prefix}<mark style='background-color:#fff2a8'>{body}</mark>{suffix}…",
                    unsafe_allow_html=True,
                )

        st.divider()
        st.subheader("청킹 (RAG 입력용)")
        mode = st.selectbox("병합 모드", ["article_only", "article±1"], index=1)
        chunks = make_chunks(raw_text, result, mode=mode)
        st.write(f"생성된 청크: {len(chunks)}개")

        with st.expander("청크 미리보기(JSON)", expanded=False):
            st.code(json.dumps([c.model_dump() for c in chunks[:5]], ensure_ascii=False, indent=2))

        # 다운로드 파일 생성 (nodes는 텍스트 제외라 매우 작습니다)
        out_dir = Path("out")
        out_dir.mkdir(exist_ok=True)
        jsonl_nodes = out_dir / "nodes.jsonl"
        jsonl_chunks = out_dir / "chunks.jsonl"
        preview_html = out_dir / "preview.html"
        zip_path = out_dir / "bundle.zip"

        to_jsonl(result.nodes, jsonl_nodes)
        to_jsonl(chunks, jsonl_chunks)

        preview_html.write_text(
            "<html><meta charset='utf-8'><body>"
            "<h3>Thai Law Parser Preview</h3>"
            f"<pre>{json.dumps([n.model_dump() for n in result.nodes[:30]], ensure_ascii=False, indent=2)}</pre>"
            "</body></html>",
            encoding="utf-8",
        )

        make_zip_bundle(zip_path, {
            "nodes.jsonl": jsonl_nodes.read_bytes(),
            "chunks.jsonl": jsonl_chunks.read_bytes(),
            "preview.html": preview_html.read_bytes(),
        })

        st.download_button("⬇️ nodes.jsonl 다운로드", data=jsonl_nodes.read_bytes(),
                           file_name="nodes.jsonl", mime="application/jsonl")
        st.download_button("⬇️ chunks.jsonl 다운로드", data=jsonl_chunks.read_bytes(),
                           file_name="chunks.jsonl", mime="application/jsonl")
        st.download_button("⬇️ 번들 ZIP 다운로드", data=zip_path.read_bytes(),
                           file_name="thai-law-parser-bundle.zip", mime="application/zip")
