import json
from pathlib import Path
import streamlit as st

from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks
from parser_core.schema import ParseResult, Node
from exporters.writers import to_jsonl, make_zip_bundle, make_debug_report

st.set_page_config(page_title="Thai Law Parser (Test)", layout="wide")
st.title("📜 Thai Law Parser — Test")

with st.sidebar:
    st.markdown("**Flow:** Upload → Parse → Review → Download")
    st.caption("v0.4 — anchored headers, prologue, scoped-duplicate check, collapsible tree, debug report")

uploaded = st.file_uploader("Upload Thai legal text (.txt)", type=["txt"])

# session
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None
    st.session_state.result: ParseResult | None = None
    st.session_state.selected_node_id: str | None = None

if uploaded:
    raw = uploaded.read().decode("utf-8", errors="ignore")
    st.session_state.raw_text = raw
    st.session_state.result = None
    st.session_state.selected_node_id = None

if st.session_state.raw_text:
    raw_text = st.session_state.raw_text

    # Scrollable raw view
    with st.expander("Raw text (scrollable)", expanded=False):
        st.markdown(
            f"""
<div style="max-height: 420px; overflow-y: auto; padding: 8px; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;">
<pre style="white-space: pre-wrap; word-break: break-word; margin: 0;">{(raw_text[:300000]).replace('<','&lt;').replace('>','&gt;')}</pre>
</div>
""",
            unsafe_allow_html=True,
        )

    auto_type = detect_doc_type(raw_text)
    st.write(f"🔎 Detected doc type: **{auto_type}**")

    forced_type = st.selectbox(
        "Doc type (override if needed)",
        options=["auto", "code", "act", "royal_decree", "regulation"],
        index=0,
        help="If detection is wrong, select the right one.",
    )
    if forced_type == "auto":
        forced_type = None

    if st.button("🧩 Run parser", use_container_width=True):
        with st.spinner("Parsing..."):
            result = parse_document(raw_text, forced_type=forced_type)
            st.session_state.result = result
            # default selection: first node
            st.session_state.selected_node_id = result.nodes[0].node_id if result.nodes else None

    result: ParseResult | None = st.session_state.result
    if result:
        st.success(f"Parsed: {len(result.nodes)} nodes, leaves {result.stats.get('leaf_count', 0)}")

        # Validation (now scoped by parent → no false duplicate for numbering restarts)
        issues = validate_tree(result)
        with st.expander(f"Consistency check (issues: {len(issues)})", expanded=False):
            if not issues:
                st.write("No issues ✅")
            else:
                for i, iss in enumerate(issues, 1):
                    st.write(f"{i}. [{iss.level}] {iss.message}")

        # layout
        left, right = st.columns([1, 2], gap="large")

        # ---- Hierarchy with collapsible expanders ----
        with left:
            st.subheader("Hierarchy (scroll & toggle)")
            st.caption("Click a node to highlight its full span on the right.")

            def render_tree(node: Node, depth: int = 0):
                # One-line label without runaway indentation for same-level peers
                label = f"{node.level} {node.label}".strip()

                # expander for any node that has children
                if node.children:
                    exp = st.expander(label, expanded=False)
                    with exp:
                        # select button for parent → highlight full span
                        if st.button(f"Select — {label}", key=f"sel-{node.node_id}"):
                            st.session_state.selected_node_id = node.node_id
                        # children
                        for ch in node.children:
                            render_tree(ch, depth + 1)
                else:
                    # leaf as simple button
                    if st.button(label, key=f"sel-{node.node_id}"):
                        st.session_state.selected_node_id = node.node_id

            # Render each root; the column itself scrolls via page, but this keeps UI simple/stable.
            for root in result.root_nodes:
                render_tree(root)

        # ---- Highlighted preview ----
        with right:
            st.subheader("Highlighted preview (scrollable)")
            sel_id = st.session_state.selected_node_id or (result.nodes[0].node_id if result.nodes else None)
            target = next((n for n in result.nodes if n.node_id == sel_id), None)
            if target:
                start, end = target.span.start, target.span.end
                start_ctx = max(0, start - 200)
                end_ctx = min(len(raw_text), end + 200)
                prefix = raw_text[start_ctx:start]
                body = raw_text[start:end]
                suffix = raw_text[end:end_ctx]
                st.markdown(
                    f"""
<div style="max-height: 480px; overflow-y: auto; padding: 8px; border: 1px solid #ddd; border-radius: 6px; background: #ffffff;">
<pre style="white-space: pre-wrap; word-break: break-word; margin: 0;">…{prefix.replace('<','&lt;').replace('>','&gt;')}<mark style="background-color:#fff2a8">{body.replace('<','&lt;').replace('>','&gt;')}</mark>{suffix.replace('<','&lt;').replace('>','&gt;')}…</pre>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Select a node on the left to preview.")

        st.divider()
        st.subheader("Chunking (for RAG)")
        mode = st.selectbox("Merge mode", ["article_only", "article±1"], index=1)
        chunks = make_chunks(raw_text, result, mode=mode)
        st.write(f"Generated chunks: {len(chunks)}")

        with st.expander("Sample chunks (JSON)", expanded=False):
            st.code(json.dumps([c.model_dump() for c in chunks[:5]], ensure_ascii=False, indent=2))

        # ---- Downloads ----
        out_dir = Path("out")
        out_dir.mkdir(exist_ok=True)
        jsonl_nodes = out_dir / "nodes.jsonl"
        jsonl_chunks = out_dir / "chunks.jsonl"
        preview_html = out_dir / "preview.html"
        zip_path = out_dir / "bundle.zip"
        debug_json = out_dir / "debug.json"

        to_jsonl(result.nodes, jsonl_nodes)
        to_jsonl(chunks, jsonl_chunks)
        preview_html.write_text(
            "<html><meta charset='utf-8'><body>"
            "<h3>Thai Law Parser Preview</h3>"
            f"<pre>{json.dumps([n.model_dump() for n in result.nodes[:30]], ensure_ascii=False, indent=2)}</pre>"
            "</body></html>",
            encoding="utf-8",
        )
        debug_report = make_debug_report(raw_text, result, chunks)
        debug_json.write_text(json.dumps(debug_report, ensure_ascii=False, indent=2), encoding="utf-8")

        make_zip_bundle(
            zip_path,
            {
                "nodes.jsonl": jsonl_nodes.read_bytes(),
                "chunks.jsonl": jsonl_chunks.read_bytes(),
                "preview.html": preview_html.read_bytes(),
                "debug.json": debug_json.read_bytes(),
            },
        )

        st.download_button("⬇️ Download nodes.jsonl", data=jsonl_nodes.read_bytes(),
                           file_name="nodes.jsonl", mime="application/jsonl")
        st.download_button("⬇️ Download chunks.jsonl", data=jsonl_chunks.read_bytes(),
                           file_name="chunks.jsonl", mime="application/jsonl")
        st.download_button("⬇️ Download bundle.zip", data=zip_path.read_bytes(),
                           file_name="thai-law-parser-bundle.zip", mime="application/zip")
        st.download_button("🐞 Download debug.json", data=debug_json.read_bytes(),
                           file_name="debug.json", mime="application/json")
