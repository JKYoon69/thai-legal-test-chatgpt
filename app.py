# app.py — Thai Law Parser (Streamlit Cloud–safe UI)
import hashlib
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# ---- your local modules (unchanged) ----
from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks
from parser_core.schema import ParseResult, Node
from exporters.writers import to_jsonl, make_zip_bundle, make_debug_report

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Thai Law Parser — Test", layout="wide")
st.title("📜 Thai Law Parser — Test")

# ------------------- GLOBAL (SAFE) CSS -------------------
# NOTE: don't style all .stButton globally except minimal pill look; tree-specific styles are scoped.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
:root { --thai-font: 'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; }

/* raw text */
.rawbox { max-height: 420px; overflow-y:auto; padding:10px;
  border:1px solid #333; border-radius:8px; background:#0e1117; }
.raw { font-family: var(--thai-font); color:#e6e6e6; white-space:pre-wrap; margin:0; }

/* gentle pill look for buttons (kept minimal to avoid layout issues on Cloud) */
.stButton > button {
  font-family: var(--thai-font);
  border-radius: 9999px; background:#1b1e23; border:1px solid #3a3a3a; color:#e6e6e6;
  padding:6px 12px;
}
.stButton > button:hover { border-color:#6ea8fe; color:#dbe9ff; }

/* tree container (left) — we scope any extra styles inside .hi-tree to avoid global leaks */
.hi-tree { max-height: 680px; overflow-y:auto; padding:6px 4px; border-right:1px solid #333; }
.hi-row  { display:flex; align-items:center; gap:10px; margin:8px 0; }

/* full document (right) */
.docwrap { width:100%; }
.docbox { max-height: 760px; overflow-y:auto; padding:16px;
  border:1px solid #333; border-radius:10px; background:#0e1117; width:100%; }
.doc { font-family:'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif;
  color:#e6e6e6; line-height:1.95; font-size:1.06rem; white-space:pre-wrap; overflow-wrap:anywhere; margin:0; }
.hlY { background:#3a3413; color:#ffe169; }   /* yellow highlight */
.hlG { background:#133a1a; color:#a7f3d0; }   /* green highlight  */
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR: CACHE CONTROL (Cloud) -------------------
with st.sidebar:
    st.markdown("**Utilities**")
    if st.button("♻️ Clear data/resource cache"):
        # Clear both modern caches; helpful on Streamlit Cloud when code/UI is iterated quickly
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.success("Cleared cache. Rerunning…")
        st.experimental_rerun()

# ------------------- SESSION STATE -------------------
ss = st.session_state
ss.setdefault("raw_text", None)
ss.setdefault("upload_sig", None)
ss.setdefault("result", None)
ss.setdefault("selected_node_id", None)
ss.setdefault("expanded", {})  # node_id -> bool

def _file_sig(file) -> str:
    data = file.getbuffer()
    h = hashlib.sha256(); h.update(data)
    return h.hexdigest()

# ------------------- UPLOAD -------------------
uploaded = st.file_uploader("Upload Thai legal text (.txt)", type=["txt"])
if uploaded is not None:
    sig = _file_sig(uploaded)
    if ss.upload_sig != sig:
        ss.upload_sig = sig
        ss.raw_text = uploaded.read().decode("utf-8", errors="ignore")
        ss.result = None
        ss.selected_node_id = None
        ss.expanded = {}

if not ss.raw_text:
    st.info("Upload a .txt file to begin.")
    st.stop()

raw_text = ss.raw_text

# ------------------- RAW TEXT BOX -------------------
with st.expander("Raw text (scrollable)", expanded=False):
    safe = raw_text[:300000].replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"<div class='rawbox'><pre class='raw'>{safe}</pre></div>", unsafe_allow_html=True)

# ------------------- PARSE CONTROLS -------------------
auto_type = detect_doc_type(raw_text)
st.write(f"🔎 Detected doc type: **{auto_type}**")
dtype = st.selectbox("Doc type (override if needed)", ["auto","code","act","royal_decree","regulation"], index=0)
if dtype == "auto": dtype = None

run_col, dl_col = st.columns([1,2])
with run_col:
    if st.button("🧩 Run parser (parse/refresh)", use_container_width=True):
        with st.spinner("Parsing..."):
            res = parse_document(raw_text, forced_type=dtype)
            ss.result = res
            ss.selected_node_id = res.nodes[0].node_id if res.nodes else None
            ss.expanded = {}

result: ParseResult | None = ss.result
if not result:
    st.stop()

# ------------------- DOWNLOADS (TOP) -------------------
with dl_col:
    out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
    nodes_p   = out_dir/"nodes.jsonl"
    chunks_p  = out_dir/"chunks.jsonl"
    preview_p = out_dir/"preview.html"
    debug_p   = out_dir/"debug.json"
    zip_p     = out_dir/"bundle.zip"

    to_jsonl(result.nodes, nodes_p)
    chunks_default = make_chunks(raw_text, result, mode="article±1")
    to_jsonl(chunks_default, chunks_p)

    preview_p.write_text(
        "<html><meta charset='utf-8'><body style=\"font-family:'Noto Sans Thai',sans-serif\">"
        f"<pre>{json.dumps([n.model_dump() for n in result.nodes[:30]], ensure_ascii=False, indent=2)}</pre>"
        "</body></html>", encoding="utf-8"
    )
    debug_p.write_text(json.dumps(make_debug_report(raw_text, result, chunks_default),
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    make_zip_bundle(zip_p, {
        "nodes.jsonl":   nodes_p.read_bytes(),
        "chunks.jsonl":  chunks_p.read_bytes(),
        "preview.html":  preview_p.read_bytes(),
        "debug.json":    debug_p.read_bytes(),
    })

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.download_button("⬇️ nodes.jsonl",  data=nodes_p.read_bytes(),  file_name="nodes.jsonl", mime="application/jsonl")
    with c2: st.download_button("⬇️ chunks.jsonl", data=chunks_p.read_bytes(), file_name="chunks.jsonl", mime="application/jsonl")
    with c3: st.download_button("⬇️ bundle.zip",   data=zip_p.read_bytes(),   file_name="thai-law-parser-bundle.zip", mime="application/zip")
    with c4: st.download_button("🐞 debug.json",   data=debug_p.read_bytes(), file_name="debug.json", mime="application/json")

st.success(f"Parsed: {len(result.nodes)} nodes, leaves {result.stats.get('leaf_count',0)}")
issues = validate_tree(result)
with st.expander(f"Consistency check (issues: {len(issues)})", expanded=False):
    if issues:
        st.write("\n".join([f"[{i.level}] {i.message}" for i in issues]))
    else:
        st.write("No issues ✅")

# ------------------- BUILD FLAT TREE -------------------
flat = []
parents = {}
def walk(n: Node, depth:int=0, parent:str|None=None):
    flat.append({"id": n.node_id, "label": n.label, "span": (n.span.start, n.span.end),
                 "depth": depth, "has_children": bool(n.children)})
    parents[n.node_id] = parent
    for ch in n.children:
        walk(ch, depth+1, n.node_id)

for r in result.root_nodes:
    walk(r, 0, None)

by_id = {x["id"]: x for x in flat}

# ------------------- LAYOUT -------------------
left, right = st.columns([1, 3.5], gap="large")  # widen right area for full doc

with left:
    st.subheader("Hierarchy")
    st.caption("Expand with ▸, collapse with ▾. Click a label to highlight its range on the right.")
    st.markdown("<div class='hi-tree'>", unsafe_allow_html=True)

    def render_node(node_id: str):
        item = by_id[node_id]
        depth = item["depth"]; has_children = item["has_children"]
        expanded = ss.expanded.get(node_id, False)
        arrow = "▾" if expanded else ("▸" if has_children else "•")

        c1, c2 = st.columns([0.14, 0.86])
        with c1:
            st.markdown("<div class='hi-row'>", unsafe_allow_html=True)
            if has_children and st.button(arrow, key=f"tg-{node_id}"):
                ss.expanded[node_id] = not expanded
            elif not has_children:
                st.write(" ")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            indent = " " * depth  # EM space indent
            st.markdown("<div class='hi-row'>", unsafe_allow_html=True)
            if st.button(f"{indent}{item['label']}", key=f"sel-{node_id}"):
                ss.selected_node_id = node_id
            st.markdown("</div>", unsafe_allow_html=True)

        if has_children and ss.expanded.get(node_id, False):
            idx = flat.index(item) + 1
            while idx < len(flat) and flat[idx]["depth"] > depth:
                if flat[idx]["depth"] == depth + 1:
                    render_node(flat[idx]["id"])
                idx += 1

    for r in [x for x in flat if x["depth"] == 0]:
        render_node(r["id"])

    st.markdown("</div>", unsafe_allow_html=True)

def compute_target(selected_id: str) -> str:
    """If any ancestor is collapsed, highlight that ancestor; else selected child."""
    target = selected_id
    cur = selected_id
    while True:
        p = parents.get(cur)
        if p is None:
            break
        if by_id[p]["has_children"] and not ss.expanded.get(p, False):
            target = p
            break
        cur = p
    return target

with right:
    st.subheader("Full document (auto-scroll & highlight)")
    color = st.radio("Highlight color", ["Yellow","Green"], horizontal=True, index=1)
    sel = ss.selected_node_id or flat[0]["id"]
    target_id = compute_target(sel)
    node = next((n for n in result.nodes if n.node_id == target_id), None)

    if node:
        s, e  = node.span.start, node.span.end
        before = raw_text[:s].replace("<","&lt;").replace(">","&gt;")
        body   = raw_text[s:e].replace("<","&lt;").replace(">","&gt;")
        after  = raw_text[e:].replace("<","&lt;").replace(">","&gt;")
        hl_cls = "hlG" if color == "Green" else "hlY"

        # IMPORTANT: CSS is included *inside* the iframe so Cloud doesn't strip styles.
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8" />
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
body {{ margin:0; background:#0e1117; }}
.docwrap {{ width:100%; }}
.docbox  {{ max-height:760px; overflow-y:auto; padding:16px; border:1px solid #333; border-radius:10px; background:#0e1117; width:100%; }}
.doc     {{ font-family:'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; color:#e6e6e6; line-height:1.95; font-size:1.06rem;
            white-space:pre-wrap; overflow-wrap:anywhere; margin:0; }}
.hlY {{ background:#3a3413; color:#ffe169; }}
.hlG {{ background:#133a1a; color:#a7f3d0; }}
</style></head>
<body>
<div class="docwrap">
  <div id="docbox" class="docbox">
    <pre class="doc">{before}<a id="SEL"></a><span class="{hl_cls}">{body}</span>{after}</pre>
  </div>
</div>
<script> const el = document.getElementById("SEL"); if (el) el.scrollIntoView({{block:'center'}}); </script>
</body></html>"""
        # width=0 → take column width 100%; height must be explicit (iframe default is 150px).
        components.html(html, height=760, width=0, scrolling=False)
    else:
        st.info("Select a node on the left to preview.")

# ------------------- CHUNKING (RAG) -------------------
st.divider()
st.subheader("Chunking (for RAG)")
mode = st.selectbox("Merge mode", ["article_only","article±1"], index=1)
chunks2 = make_chunks(raw_text, ss.result, mode=mode)
st.write(f"Generated chunks: {len(chunks2)}")
with st.expander("Sample chunks (JSON)", expanded=False):
    st.code(json.dumps([c.model_dump() for c in chunks2[:5]], ensure_ascii=False, indent=2))
