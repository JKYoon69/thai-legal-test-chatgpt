import hashlib
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks
from parser_core.schema import ParseResult, Node
from exporters.writers import to_jsonl, make_zip_bundle, make_debug_report

st.set_page_config(page_title="Thai Law Parser (Test)", layout="wide")
st.title("📜 Thai Law Parser — Test")

# ---- Global font/style (Thai-safe) ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
:root { --thai-font: 'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; }
.thai-box { font-family: var(--thai-font); line-height: 1.75; font-size: 1rem; color: #e6e6e6; white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; }
.hl { background-color:#fff2a8; color:#111; }
.box { max-height: 600px; overflow-y: auto; padding: 8px; border: 1px solid #ddd; border-radius: 6px; background: #0e1117; }
.leftpane { max-height: 600px; overflow-y: auto; padding-right: 6px; border: 1px solid #333; border-radius: 6px; }
.node-btn { width: 100%; text-align: left; padding: 6px 10px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; color: #e6e6e6; margin: 2px 0; }
.node-btn.sel { background: #2d3a4a; border-color:#5b8aaa; }
.meta { color:#9aa0a6; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Flow:** Upload → Parse → Review → Download")
    st.caption("v0.8 — front_matter grouping, full-doc preview with auto-scroll")

# ---- session state ----
ss = st.session_state
ss.setdefault("raw_text", None)
ss.setdefault("upload_sig", None)
ss.setdefault("result", None)
ss.setdefault("selected_node_id", None)

def _file_sig(file):
    data = file.getbuffer()
    h = hashlib.sha256(); h.update(data)
    return h.hexdigest()

uploaded = st.file_uploader("Upload Thai legal text (.txt)", type=["txt"], key="uploader")

if uploaded is not None:
    sig = _file_sig(uploaded)
    if ss.upload_sig != sig:
        raw = uploaded.read().decode("utf-8", errors="ignore")
        ss.raw_text = raw
        ss.upload_sig = sig
        ss.result = None
        ss.selected_node_id = None

if ss.raw_text:
    raw_text = ss.raw_text

    with st.expander("Raw text (scrollable)", expanded=False):
        safe = (raw_text[:300000]).replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f"<div class='box'><pre class='thai-box'>{safe}</pre></div>", unsafe_allow_html=True)

    auto_type = detect_doc_type(raw_text)
    st.write(f"🔎 Detected doc type: **{auto_type}**")

    forced_type = st.selectbox(
        "Doc type (override if needed)",
        options=["auto", "code", "act", "royal_decree", "regulation"],
        index=0,
        help="If detection is wrong, select the right one.",
        key="doc_type_select",
    )
    if forced_type == "auto":
        forced_type = None

    if st.button("🧩 Run parser", use_container_width=True, key="run_parser"):
        with st.spinner("Parsing..."):
            result = parse_document(raw_text, forced_type=forced_type)
            ss.result = result
            ss.selected_node_id = result.nodes[0].node_id if result.nodes else None

    result: ParseResult | None = ss.result
    if result:
        st.success(f"Parsed: {len(result.nodes)} nodes, leaves {result.stats.get('leaf_count', 0)}")

        issues = validate_tree(result)
        with st.expander(f"Consistency check (issues: {len(issues)})", expanded=False):
            if not issues:
                st.write("No issues ✅")
            else:
                for i, iss in enumerate(issues, 1):
                    st.write(f"{i}. [{iss.level}] {iss.message}")

        # Build flat list
        flat: list[dict] = []
        def walk(n: Node, depth:int=0):
            flat.append({"id": n.node_id, "label": n.label, "span": (n.span.start, n.span.end), "depth": depth})
            for ch in n.children:
                walk(ch, depth+1)
        for root in result.root_nodes:
            walk(root, 0)

        left, right = st.columns([1, 2], gap="large")

        # ---- LEFT: full tree list (scrollable) with auto-scroll to selected ----
        with left:
            st.subheader("Hierarchy")
            st.caption("Click a node; the right pane scrolls to that section.")

            st.markdown("<div id='leftpane' class='leftpane'>", unsafe_allow_html=True)
            for item in flat:
                indent = "&nbsp;" * (item["depth"] * 4)
                sel_cls = " sel" if item["id"] == (ss.selected_node_id or "") else ""
                # anchor before button for JS scroll
                st.markdown(f"<div id='li-{item['id']}'></div>", unsafe_allow_html=True)
                with st.form(key=f"frm-{item['id']}"):
                    st.markdown(f"{indent}<button class='node-btn{sel_cls}' type='submit'>{item['label']}</button>", unsafe_allow_html=True)
                    submitted = st.form_submit_button("", use_container_width=False)
                    if submitted:
                        ss.selected_node_id = item["id"]
            st.markdown("</div>", unsafe_allow_html=True)

            # auto-scroll LEFT to selected
            if ss.selected_node_id:
                components.html(f"""
                    <script>
                    const el = parent.document.getElementById("li-{ss.selected_node_id}");
                    if (el) el.scrollIntoView({{block:'center', behavior:'instant'}});
                    </script>""", height=0)

        # ---- RIGHT: full document, auto-scroll + highlight selection ----
        with right:
            st.subheader("Full document (auto-scroll & highlight)")
            sel_id = ss.selected_node_id or (flat[0]["id"] if flat else None)
            target = next((n for n in result.nodes if n.node_id == sel_id), None)

            if target:
                s, e = target.span.start, target.span.end
                # build full HTML once with one anchor+highlight
                before = raw_text[:s].replace("<","&lt;").replace(">","&gt;")
                body   = raw_text[s:e].replace("<","&lt;").replace(">","&gt;")
                after  = raw_text[e:].replace("<","&lt;").replace(">","&gt;")

                html = f"""
<div id="docbox" class="box">
  <pre class="thai-box">{before}<a id="SEL"></a><span class="hl">{body}</span>{after}</pre>
</div>
<script>
  const sel = document.getElementById("SEL");
  if (sel) sel.scrollIntoView({{block:'center'}});
</script>
"""
                components.html(html, height=600, scrolling=False)
            else:
                st.info("Select a node on the left to preview.")

        st.divider()
        st.subheader("Chunking (for RAG)")
        mode = st.selectbox("Merge mode", ["article_only", "article±1"], index=1, key="merge_mode")
        chunks = make_chunks(raw_text, result, mode=mode)
        st.write(f"Generated chunks: {len(chunks)}")

        with st.expander("Sample chunks (JSON)", expanded=False):
            st.code(json.dumps([c.model_dump() for c in chunks[:5]], ensure_ascii=False, indent=2))

        # ---- Downloads ----
        out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
        jsonl_nodes = out_dir / "nodes.jsonl"
        jsonl_chunks = out_dir / "chunks.jsonl"
        preview_html = out_dir / "preview.html"
        zip_path = out_dir / "bundle.zip"
        debug_json = out_dir / "debug.json"

        to_jsonl(result.nodes, jsonl_nodes)
        to_jsonl(chunks, jsonl_chunks)
        preview_html.write_text(
            "<html><meta charset='utf-8'><body style=\"font-family:'Noto Sans Thai',sans-serif\">"
            "<h3>Thai Law Parser Preview</h3>"
            f"<pre>{json.dumps([n.model_dump() for n in result.nodes[:30]], ensure_ascii=False, indent=2)}</pre>"
            "</body></html>",
            encoding="utf-8",
        )
        debug_report = make_debug_report(raw_text, result, chunks)
        debug_json.write_text(json.dumps(debug_report, ensure_ascii=False, indent=2), encoding="utf-8")

        make_zip_bundle(zip_path, {
            "nodes.jsonl": jsonl_nodes.read_bytes(),
            "chunks.jsonl": jsonl_chunks.read_bytes(),
            "preview.html": preview_html.read_bytes(),
            "debug.json": debug_json.read_bytes(),
        })

        st.download_button("⬇️ Download nodes.jsonl", data=jsonl_nodes.read_bytes(),
                           file_name="nodes.jsonl", mime="application/jsonl", key="dl_nodes")
        st.download_button("⬇️ Download chunks.jsonl", data=jsonl_chunks.read_bytes(),
                           file_name="chunks.jsonl", mime="application/jsonl", key="dl_chunks")
        st.download_button("⬇️ Download bundle.zip", data=zip_path.read_bytes(),
                           file_name="thai-law-parser-bundle.zip", mime="application/zip", key="dl_zip")
        st.download_button("🐞 Download debug.json", data=debug_json.read_bytes(),
                           file_name="debug.json", mime="application/json", key="dl_debug")
