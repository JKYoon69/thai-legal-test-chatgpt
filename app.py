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

# =================== Global style (text-only tree, dark previews) ===================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
:root { --thai-font: 'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; }

/* Raw text (dark) */
.rawbox { max-height: 420px; overflow-y: auto; padding: 10px;
  border: 1px solid #333; border-radius: 8px; background: #0e1117; }
.raw { font-family: var(--thai-font); color:#e6e6e6; white-space: pre-wrap; margin: 0; }

/* Tree: text-only (remove button chrome completely) */
.tree { max-height: 640px; overflow-y: auto; padding: 2px 4px; border-right: 1px solid #333; }
.tree .row { display:flex; align-items:center; gap:6px; margin: 2px 0; }
.tree .indent { width:0; }
.tree .stButton > button {
  background: transparent !important; border: none !important; box-shadow:none !important;
  padding: 0 !important; margin: 0 !important; border-radius: 0 !important;
  color: #e6e6e6 !important; font-family: var(--thai-font); font-size: 0.98rem; text-align:left;
}
.tree .chev .stButton > button { width: 20px; text-align:center; }
.tree .label .stButton > button:hover { color:#ffe169 !important; }

/* Full document: dark bg + white text; highlighted text in yellow/green */
.docbox { max-height: 640px; overflow-y: auto; padding: 12px;
  border: 1px solid #333; border-radius: 8px; background: #0e1117; width: 100%; }
.doc { font-family: var(--thai-font); color:#e6e6e6; line-height:1.9; font-size:1.05rem;
  white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; margin:0; }
.hlY { background:#3a3413; color:#ffe169; }  /* yellow */
.hlG { background:#133a1a; color:#a7f3d0; }  /* green  */
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Flow:** Upload → Parse → Review → Download")
    st.caption("UI v4 — pure-text tree, correct parent/child highlight, responsive doc")

# =================== Session ===================
ss = st.session_state
ss.setdefault("raw_text", None)
ss.setdefault("upload_sig", None)
ss.setdefault("result", None)
ss.setdefault("selected_node_id", None)
ss.setdefault("expanded", {})  # node_id -> bool
ss.setdefault("hl_color", "Yellow")      # highlight color selector

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
        ss.expanded = {}

if not ss.raw_text:
    st.info("Upload a .txt file to begin.")
    st.stop()

raw_text = ss.raw_text

# 1) Raw text (scrollable) — dark background
with st.expander("Raw text (scrollable)", expanded=False):
    safe = (raw_text[:300000]).replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"<div class='rawbox'><pre class='raw'>{safe}</pre></div>", unsafe_allow_html=True)

# Controls (Run + Downloads near top)
auto_type = detect_doc_type(raw_text)
st.write(f"🔎 Detected doc type: **{auto_type}**")
forced_type = st.selectbox("Doc type (override if needed)",
                           ["auto","code","act","royal_decree","regulation"], index=0)
if forced_type == "auto": forced_type = None

run_col, dl_col = st.columns([1,2])
with run_col:
    run_clicked = st.button("🧩 Run parser", use_container_width=True)

if run_clicked:
    with st.spinner("Parsing..."):
        result = parse_document(raw_text, forced_type=forced_type)
        ss.result = result
        ss.selected_node_id = result.nodes[0].node_id if result.nodes else None
        ss.expanded = {}

result: ParseResult|None = ss.result
if not result:
    st.stop()

# Downloads (fixed spot under Run)
with dl_col:
    out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
    jsonl_nodes = out_dir / "nodes.jsonl"
    jsonl_chunks = out_dir / "chunks.jsonl"
    preview_html = out_dir / "preview.html"
    zip_path = out_dir / "bundle.zip"
    debug_json = out_dir / "debug.json"

    to_jsonl(result.nodes, jsonl_nodes)
    chunks_default = make_chunks(raw_text, result, mode="article±1")
    to_jsonl(chunks_default, jsonl_chunks)
    preview_html.write_text(
        "<html><meta charset='utf-8'><body style=\"font-family:'Noto Sans Thai',sans-serif\">"
        "<h3>Thai Law Parser Preview</h3>"
        f"<pre>{json.dumps([n.model_dump() for n in result.nodes[:30]], ensure_ascii=False, indent=2)}</pre>"
        "</body></html>", encoding="utf-8")
    debug_report = make_debug_report(raw_text, result, chunks_default)
    debug_json.write_text(json.dumps(debug_report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_zip_bundle(zip_path, {
        "nodes.jsonl": jsonl_nodes.read_bytes(),
        "chunks.jsonl": jsonl_chunks.read_bytes(),
        "preview.html": preview_html.read_bytes(),
        "debug.json": debug_json.read_bytes(),
    })

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.download_button("⬇️ nodes.jsonl", data=jsonl_nodes.read_bytes(),
                                file_name="nodes.jsonl", mime="application/jsonl", key="dl_nodes")
    with c2: st.download_button("⬇️ chunks.jsonl", data=jsonl_chunks.read_bytes(),
                                file_name="chunks.jsonl", mime="application/jsonl", key="dl_chunks")
    with c3: st.download_button("⬇️ bundle.zip", data=zip_path.read_bytes(),
                                file_name="thai-law-parser-bundle.zip", mime="application/zip", key="dl_zip")
    with c4: st.download_button("🐞 debug.json", data=debug_json.read_bytes(),
                                file_name="debug.json", mime="application/json", key="dl_debug")

# Status
st.success(f"Parsed: {len(result.nodes)} nodes, leaves {result.stats.get('leaf_count',0)}")
issues = validate_tree(result)
with st.expander(f"Consistency check (issues: {len(issues)})", expanded=False):
    st.write("No issues ✅" if not issues else "\n".join([f"[{i.level}] {i.message}" for i in issues]))

# =================== Build flat/depth + parent map ===================
flat: list[dict] = []
parents: dict[str, str|None] = {}  # node_id -> parent_id
def walk(n: Node, depth:int=0, parent_id:str|None=None):
    flat.append({"id": n.node_id, "label": n.label, "span": (n.span.start, n.span.end),
                 "depth": depth, "has_children": bool(n.children)})
    parents[n.node_id] = parent_id
    for ch in n.children: walk(ch, depth+1, n.node_id)
for root in result.root_nodes: walk(root, 0, None)
by_id = {x["id"]: x for x in flat}

# =================== Layout: Tree (text-only) + Full document ===================
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Hierarchy")
    st.caption("Expand/collapse with ▸ ▾. Click a node to highlight its range on the right.")
    st.markdown("<div class='tree'>", unsafe_allow_html=True)

    # recursive renderer (text-only)
    def render_node(node_id: str):
        item = by_id[node_id]
        depth = item["depth"]; has_children = item["has_children"]
        indent_px = depth * 16  # slight indent for children
        expanded = ss.expanded.get(node_id, False)
        arrow = "▾" if expanded else ("▸" if has_children else "•")

        # left chevron + label (no boxes; styled to plain text)
        cols = st.columns([0.12, 0.88])
        with cols[0]:
            if has_children:
                if st.button(arrow, key=f"tg-{node_id}"):
                    now = not expanded
                    ss.expanded[node_id] = now
                    # 접힌 순간: 부모 범위를 보이도록(요구사항) → 선택을 부모로 바꾸지 않아도
                    # 아래 highlight 계산에서 접힌 조상을 우선 사용하므로 자동 반영됨.
            else:
                st.write(" ")
        with cols[1]:
            # label as plain text button
            if st.button(item["label"], key=f"sel-{node_id}"):
                ss.selected_node_id = node_id
            # apply indent via spacer
            st.markdown(f"<div style='height:0; margin-left:{indent_px}px'></div>", unsafe_allow_html=True)

        # children (only if expanded)
        if has_children and ss.expanded.get(node_id, False):
            idx = flat.index(item) + 1
            while idx < len(flat) and flat[idx]["depth"] > depth:
                if flat[idx]["depth"] == depth + 1:
                    render_node(flat[idx]["id"])
                idx += 1

    # render all roots
    for r in [x for x in flat if x["depth"] == 0]:
        render_node(r["id"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Highlight target logic ----
def compute_highlight_target(selected_id: str) -> str:
    """If any ancestor is collapsed, highlight that ancestor; else highlight selected."""
    target_id = selected_id
    cur = selected_id
    while True:
        parent_id = parents.get(cur)
        if parent_id is None:
            break
        if by_id[parent_id]["has_children"] and not ss.expanded.get(parent_id, False):
            target_id = parent_id   # closest collapsed ancestor wins
            break
        cur = parent_id
    return target_id

# ---- RIGHT: Full document (dark, white text, highlighted selection) ----
with right:
    st.subheader("Full document (auto-scroll & highlight)")
    ss.hl_color = st.radio("Highlight color", ["Yellow","Green"], horizontal=True, index=0)
    sel_id = ss.selected_node_id or flat[0]["id"]
    target_id = compute_highlight_target(sel_id)
    target = next((n for n in result.nodes if n.node_id == target_id), None)

    if target:
        s, e = target.span.start, target.span.end
        before = raw_text[:s].replace("<","&lt;").replace(">","&gt;")
        body   = raw_text[s:e].replace("<","&lt;").replace(">","&gt;")
        after  = raw_text[e:].replace("<","&lt;").replace(">","&gt;")
        hl_cls = "hlY" if ss.hl_color == "Yellow" else "hlG"
        html = f"""
<div id="docbox" class="docbox" style="width:100%;">
  <pre class="doc">{before}<a id="SEL"></a><span class="{hl_cls}">{body}</span>{after}</pre>
</div>
<script>
  const sel = document.getElementById("SEL");
  if (sel) sel.scrollIntoView({{block:'center'}});
</script>
"""
        components.html(html, height=640, scrolling=False, width=0)  # width=0 -> container width(100%)
    else:
        st.info("Select a node on the left to preview.")

# ---- Chunking controls ----
st.divider()
st.subheader("Chunking (for RAG)")
mode = st.selectbox("Merge mode", ["article_only", "article±1"], index=1, key="merge_mode")
chunks = make_chunks(raw_text, ss.result, mode=mode)
st.write(f"Generated chunks: {len(chunks)}")
with st.expander("Sample chunks (JSON)", expanded=False):
    st.code(json.dumps([c.model_dump() for c in chunks[:5]], ensure_ascii=False, indent=2))
