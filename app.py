import hashlib
import json
from pathlib import Path
import streamlit as st

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
.thai-box {
  font-family: var(--thai-font);
  line-height: 1.75;
  font-size: 1rem;
  color: #e6e6e6;
  white-space: pre-wrap;
  /* 줄바꿈 규칙: 조합문자 깨짐을 줄이기 위해 normal 유지, 너무 긴 줄은 anywhere로 */
  word-break: normal;
  overflow-wrap: anywhere;
}
.hl { background-color:#fff2a8; color:#111; }
.box {
  max-height: 480px; overflow-y: auto; padding: 8px;
  border: 1px solid #ddd; border-radius: 6px; background: #0e1117;
}
.leftpane { max-height: 540px; overflow-y: auto; padding-right: 6px; }
.node-btn {
  width: 100%; text-align: left; padding: 6px 10px; border: 1px solid #444;
  border-radius: 8px; background: #1e1e1e; color: #e6e6e6; margin: 4px 0;
}
.node-btn.sel { background: #2d3a4a; border-color:#5b8aaa; }
.meta { color:#9aa0a6; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Flow:** Upload → Parse → Review → Download")
    st.caption("v0.7 — optional 'ที่', Thai font, virtual scroll (Flat mode)")

# ---- session state ----
ss = st.session_state
ss.setdefault("raw_text", None)
ss.setdefault("upload_sig", None)        # hash to detect new uploads
ss.setdefault("result", None)
ss.setdefault("selected_node_id", None)
ss.setdefault("expanded", {})            # node_id -> bool
ss.setdefault("view_mode", "Flat")       # "Flat" | "Tree"

def _file_sig(file):
    data = file.getbuffer()
    h = hashlib.sha256(); h.update(data)
    return h.hexdigest()

uploaded = st.file_uploader("Upload Thai legal text (.txt)", type=["txt"], key="uploader")

# Only reset when the uploaded file actually changed
if uploaded is not None:
    sig = _file_sig(uploaded)
    if ss.upload_sig != sig:
        raw = uploaded.read().decode("utf-8", errors="ignore")
        ss.raw_text = raw
        ss.upload_sig = sig
        ss.result = None
        ss.selected_node_id = None
        ss.expanded = {}
    else:
        pass

if ss.raw_text:
    raw_text = ss.raw_text

    # Scrollable raw view with Thai font
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
            ss.expanded = {}

    result: ParseResult | None = ss.result
    if result:
        st.success(f"Parsed: {len(result.nodes)} nodes, leaves {result.stats.get('leaf_count', 0)}")

        # Validation
        issues = validate_tree(result)
        with st.expander(f"Consistency check (issues: {len(issues)})", expanded=False):
            if not issues:
                st.write("No issues ✅")
            else:
                for i, iss in enumerate(issues, 1):
                    st.write(f"{i}. [{iss.level}] {iss.message}")

        # build flat list for Flat-mode virtual scroll
        flat: list[dict] = []
        def walk(n: Node, depth:int=0):
            flat.append({"id": n.node_id, "label": n.label, "span": (n.span.start, n.span.end), "depth": depth, "has_children": bool(n.children)})
            for ch in n.children:
                walk(ch, depth+1)
        for root in result.root_nodes:
            walk(root, 0)

        left, right = st.columns([1, 2], gap="large")

        with left:
            st.subheader("Hierarchy (scroll & toggle)")
            ss.view_mode = st.radio("View mode", ["Flat", "Tree"], horizontal=True, index=0 if ss.view_mode=="Flat" else 1)

            if ss.view_mode == "Flat":
                # --- Virtual scroll window around selected ---
                sel_id = ss.selected_node_id or (flat[0]["id"] if flat else None)
                idx = next((i for i,x in enumerate(flat) if x["id"]==sel_id), 0)
                window = 30
                start = max(0, idx - window//2)
                end = min(len(flat), start + window)
                st.caption(f"Showing {start+1}–{end} / {len(flat)}")

                st.markdown("<div class='leftpane'>", unsafe_allow_html=True)
                for i in range(start, end):
                    item = flat[i]
                    indent = "&nbsp;" * (item["depth"] * 4)
                    sel_cls = " sel" if item["id"] == sel_id else ""
                    # use form to avoid duplicate key issues on rerun
                    with st.form(key=f"frm-{item['id']}"):
                        st.markdown(f"{indent}<button class='node-btn{sel_cls}' type='submit'>{item['label']}</button>", unsafe_allow_html=True)
                        submitted = st.form_submit_button("", use_container_width=False)
                        if submitted:
                            ss.selected_node_id = item["id"]
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                # --- Tree mode with expand/collapse ---
                st.caption("Click a label to select; parents toggle open/close.")

                def render_tree(nodes: list[Node], depth: int = 0):
                    for node in nodes:
                        label = node.label
                        has_children = bool(node.children)
                        # toggle
                        if has_children:
                            toggled = st.checkbox("", key=f"ex-{node.node_id}",
                                                  value=ss.expanded.get(node.node_id, False), help="Expand/collapse")
                            ss.expanded[node.node_id] = toggled
                        else:
                            st.write("")
                        # select
                        if st.button(label, key=f"sel-{node.node_id}"):
                            ss.selected_node_id = node.node_id
                        # children
                        if has_children and ss.expanded.get(node.node_id, False):
                            render_tree(node.children, depth+1)

                render_tree(result.root_nodes, depth=0)

        # ---- Highlighted preview ----
        with right:
            st.subheader("Highlighted preview (scrollable)")
            sel_id = ss.selected_node_id or (flat[0]["id"] if flat else None)
            target = next((n for n in result.nodes if n.node_id == sel_id), None)
            if target:
                s, e = target.span.start, target.span.end
                s_ctx = max(0, s - 200); e_ctx = min(len(raw_text), e + 200)
                pre = raw_text[s_ctx:s]; body = raw_text[s:e]; suf = raw_text[e:e_ctx]
                pre = pre.replace("<","&lt;").replace(">","&gt;")
                body = body.replace("<","&lt;").replace(">","&gt;")
                suf = suf.replace("<","&lt;").replace(">","&gt;")
                st.markdown(
                    f"<div class='box'><pre class='thai-box'>…{pre}<span class='hl'>{body}</span>{suf}…</pre></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Select a node on the left to preview.")

        st.divider()
        st.subheader("Chunking (for RAG)")
        mode = st.selectbox("Merge mode", ["article_only", "article±1"], index=1, key="merge_mode")
        chunks = make_chunks(raw_text, result, mode=mode)
        st.write(f"Generated chunks: {len(chunks)}")

        with st.expander("Sample chunks (JSON)", expanded=False):
            st.code(json.dumps([c.model_dump() for c in chunks[:5]], ensure_ascii=False, indent=2))

        # ---- Downloads (state persists) ----
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
