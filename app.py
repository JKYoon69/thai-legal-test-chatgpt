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

# =============== Global style ===============
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
:root { --thai-font: 'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; }

/* 1) 모든 Streamlit 버튼을 "텍스트만"으로 보이게 (트리 라벨/토글용) */
div.stButton > button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  padding: 2px 0 !important; margin: 0 !important; border-radius: 0 !important;
  color: #e6e6e6 !important; font-family: var(--thai-font); font-size: 0.98rem; text-align: left;
}

/* 2) Raw text (dark) */
.rawbox { max-height: 420px; overflow-y: auto; padding: 10px;
  border: 1px solid #333; border-radius: 8px; background: #0e1117; }
.raw { font-family: var(--thai-font); color:#e6e6e6; white-space: pre-wrap; margin: 0; }

/* 3) Hierarchy: 텍스트 리스트 */
.tree { max-height: 640px; overflow-y: auto; padding: 2px 4px; border-right: 1px solid #333; }
.tree .row { display:flex; align-items:center; gap:8px; margin: 4px 0; }
.tree .arrow { width: 22px; text-align:center; color:#e6e6e6; }
.tree .label { color:#e6e6e6; }
.tree .label:hover { color:#ffe169; }
.tree .indent { display:inline-block; }

/* 4) Full document (dark bg + white text); highlight */
.docbox { max-height: 640px; overflow-y: auto; padding: 12px;
  border: 1px solid #333; border-radius: 8px; background: #0e1117; width: 100%; }
.doc { font-family: var(--thai-font); color:#e6e6e6; line-height:1.9; font-size:1.05rem;
  white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; margin:0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Flow:** Upload → Parse → Review → Download")
    st.caption("UI v5 — text-only tree, correct parent/child highlight, dark full-doc")

# =============== Session ===============
ss = st.session_state
ss.setdefault("raw_text", None)
ss.setdefault("upload_sig", None)
ss.setdefault("result", None)
ss.setdefault("selected_node_id", None)
ss.setdefault("expanded", {})      # node_id -> bool
ss.setdefault("hl_color", "Yellow")

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

# ---- Raw text (scrollable) on dark background ----
with st.expander("Raw text (scrollable)", expanded=False):
    safe = (raw_text[:300000]).replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"<div class='rawbox'><pre class='raw'>{safe}</pre></div>", unsafe_allow_html=True)

# ---- Controls ----
auto_type = detect_doc_type(raw_text)
st.write(f"🔎 Detected doc type: **{auto_type}**")
forced_type = st.selectbox("Doc type (override if needed)",
                           ["auto","code","act","royal_decree","regulation"], index=0)
if forced_type == "auto": forced_type = None

run_col, dl_col = st.columns([1,2])
with run_col:
    run_clicked = st.button("🧩 Run parser (parse/refresh)", use_container_width=True)

if run_clicked:
    with st.spinner("Parsing..."):
        result = parse_document(raw_text, forced_type=forced_type)
        ss.result = result
        ss.selected_node_id = result.nodes[0].node_id if result.nodes else None
        ss.expanded = {}

result: ParseResult|None = ss.result
if not result:
    st.stop()

# ---- Downloads placed near the top ----
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

st.success(f"Parsed: {len(result.nodes)} nodes, leaves {result.stats.get('leaf_count',0)}")
issues = validate_tree(result)
with st.expander(f"Consistency check (issues: {len(issues)})", expanded=False):
    st.write("No issues ✅" if not issues else "\n".join([f"[{i.level}] {i.message}" for i in issues]))

# =============== Build flat + parent map ===============
flat: list[dict] = []
parents: dict[str, str|None] = {}
def walk(n: Node, depth:int=0, parent_id:str|None=None):
    flat.append({"id": n.node_id, "label": n.label, "span": (n.span.start, n.span.end),
                 "depth": depth, "has_children": bool(n.children)})
    parents[n.node_id] = parent_id
    for ch in n.children: walk(ch, depth+1, n.node_id)
for root in result.root_nodes: walk(root, 0, None)
by_id = {x["id"]: x for x in flat}

# =============== Layout ===============
left, right = st.columns([1, 2], gap="large")

# ---- LEFT: text-only expandable tree ----
with left:
    st.subheader("Hierarchy")
    st.caption("Expand with ▸, collapse with ▾. Click a label to highlight its range on the right.")
    st.markdown("<div class='tree'>", unsafe_allow_html=True)

    def render_node(node_id: str):
        item = by_id[node_id]
        depth = item["depth"]; has_children = item["has_children"]
        expanded = ss.expanded.get(node_id, False)
        arrow = "▾" if expanded else ("▸" if has_children else "•")

        # 한 줄 구성: [arrow] [indent+label]
        colA, colB = st.columns([0.1, 0.9])
        with colA:
            if has_children:
                if st.button(arrow, key=f"tg-{node_id}"):
                    ss.expanded[node_id] = not expanded
