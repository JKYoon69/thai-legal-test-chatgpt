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

# =================== Global style ===================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
:root { --thai-font: 'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; }

/* Raw text (dark) */
.rawbox { max-height: 420px; overflow-y: auto; padding: 10px;
  border: 1px solid #333; border-radius: 8px; background: #0e1117; }
.raw { font-family: var(--thai-font); color:#e6e6e6; white-space: pre-wrap; margin: 0; }

/* Left tree: text-only (no boxes around buttons) */
.tree { max-height: 640px; overflow-y: auto; padding: 6px 4px; border-right: 1px solid #333; }
.tree .row { display:flex; align-items:center; gap:6px; margin: 2px 0; }
.tree .indent { width:0; }
.tree .chev .stButton>button {
  background: transparent !important; border: none !important; color:#e6e6e6 !important;
  padding: 2px 4px !important; min-width: 22px;
  font-family: var(--thai-font); font-size: 0.95rem;
}
.tree .label .stButton>button {
  background: transparent !important; border: none !important; color:#e6e6e6 !important;
  padding: 2px 4px !important;
  font-family: var(--thai-font); font-size: 0.95rem; text-align:left;
}
.tree .label .stButton>button:hover { color:#8ab4f8 !important; }

/* Right full document: dark bg + white text; selection highlighted */
.docbox { max-height: 640px; overflow-y: auto; padding: 12px;
  border: 1px solid #333; border-radius: 8px; background: #0e1117; width: 100%; }
.doc { font-family: var(--thai-font); color:#e6e6e6; line-height:1.9; font-size:1.05rem;
  white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; margin:0; }
.hl { background:#3a3413; color:#ffe169; }   /* yellow highlight */
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Flow:** Upload → Parse → Review → Download")
    st.caption("UI v3 — pure-text tree, correct parent/child highlight logic, responsive doc")

# =================== Session ===================
ss = st.session_state
ss.setdefault("raw_text", None)
ss.setdefault("upload_sig", None)
ss.setdefault("result", None)
ss.setdefault("selected_node_id", None)
ss.setdefault("expanded", {})  # node_id -> bool

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

# 1) Raw text (scrollable) — dark background (요청 1 반영)
with st.expander("Raw text (scrollable)", expanded=False):
    safe = (raw_text[:300000]).replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"<div class='rawbox'><pre class='raw'>{safe}</pre></div>", unsafe_allow_html=True)

# Controls (Run + Downloads 상단 배치)
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

# Prepare downloads (요청 4의 “Run 바로 아래” 위치 유지)
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

# =================== Build flat with depth ===================
flat: list[dict] = []
parents: dict[str, str|None] = {}  # node_id -> parent_id
def walk(n: Node, depth:int=0, parent_id:str|None=None):
    flat.append({"id": n.node_id, "label": n.label, "span": (n.span.start, n.span.end),
                 "depth": depth, "has_children": bool(n.children)})
    parents[n.node_id] = parent_id
    for ch in n.children: walk(ch, depth+1, n.node_id)
for root in result.root_nodes: walk(root, 0, None)
by_id = {x["id"]: x for x in flat}

def is_descendant(desc_id: str, anc_id: str) -> bool:
    cur = parents.get(desc_id)
    while cur is not None:
        if cur == anc_id: return True
        cur = parents.get(cur)
    return False

# =================== Layout: Tree + Full Document ===================
left, right = st.columns([1, 2], gap="large")

# ---- LEFT: Text-only tree (요청 2의 첫 줄 반영) ----
with left:
    st.subheader("Hierarchy")
    st.caption("Expand/collapse with ▸ ▾. Click a node to highlight its range on the right.")
    st.markdown("<div class='tree'>", unsafe_allow_html=True)

    # 어떤 노드가 펼쳐졌는지에 따라 보이는 아이만 재귀적으로 그리기
    def render_node(node_id: str):
        item = by_id[node_id]
        depth = item["depth"]; has_children = item["has_children"]
        indent_px = depth * 16
        expanded = ss.expanded.get(node_id, False)
        arrow = "▾" if expanded else ("▸" if has_children else "•")

        # 한 줄: [chevron button] [label button] — 텍스트만 보이도록 스타일 지정
        row_left, row_right = st.columns([0.1, 0.9])
        with row_left:
            if has_children:
                if st.button(arrow, key=f"tg-{node_id}"):
                    # 토글 시: 접히면 오른쪽은 "부모 전체 범위"를 보이게 부모를 선택
                    now = not expanded
                    ss.expanded[node_id] = now
                    if not now:
                        ss.selected_node_id = node_id
            else:
                st.write(" ")

        with row_right:
            if st.button(item["label"], key=f"sel-{node_id}"):
                ss.selected_node_id = node_id

        # Children (펼쳐져 있을 때만)
        if has_children and ss.expanded.get(node_id, False):
            # 바로 아래 depth+1 만 순회
            idx = flat.index(item) + 1
            while idx < len(flat) and flat[idx]["depth"] > depth:
                if flat[idx]["depth"] == depth + 1:
                    render_node(flat[idx]["id"])
                idx += 1

    # 0-depth 루트부터 출력
    for root_item in [x for x in flat if x["depth"] == 0]:
        render_node(root_item["id"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---- RIGHT: Full document (요청 3 모두 반영) ----
with right:
    st.subheader("Full document (auto-scroll & highlight)")
    # 하이라이트 대상 결정 로직:
    # - 기본: ss.selected_node_id
    # - 만약 선택된 노드의 어떤 조상이라도 접혀 있다면, 그 "접힌 조상"을 우선 하이라이트
    sel = ss.selected_node_id or flat[0]["id"]
    target_id = sel
    # 접힌 조상이 있으면 가장 가까운 접힌 조상으로 대체
    cur = sel
    while cur is not None:
        p = parents.get(cur)
        if p is not None and ss.expanded.get(p, False) is False and by_id[p]["has_children"]:
            target_id = p
        cur = p

    target = next((n for n in result.nodes if n.node_id == target_id), None)
    if target:
        s, e = target.span.start, target.span.end
        before = raw_text[:s].replace("<","&lt;").replace(">","&gt;")
        body   = raw_text[s:e].replace("<","&lt;").replace(">","&gt;")
        after  = raw_text[e:].replace("<","&lt;").replace(">","&gt;")
        html = f"""
<div id="docbox" class="docbox" style="width:100%;">
  <pre class="doc">{before}<a id="SEL"></a><span class="hl">{body}</span>{after}</pre>
</div>
<script>
  const sel = document.getElementById("SEL");
  if (sel) sel.scrollIntoView({{block:'center'}});
</script>
"""
        # width=0 → container width에 맞춰 자동
        components.html(html, height=640, scrolling=False, width=0)
    else:
        st.info("Select a node on the left to preview.")

# ---- Chunking controls (existing) ----
st.divider()
st.subheader("Chunking (for RAG)")
mode = st.selectbox("Merge mode", ["article_only", "article±1"], index=1, key="merge_mode")
chunks = make_chunks(raw_text, ss.result, mode=mode)
st.write(f"Generated chunks: {len(chunks)}")

with st.expander("Sample chunks (JSON)", expanded=False):
    st.code(json.dumps([c.model_dump() for c in chunks[:5]], ensure_ascii=False, indent=2))
