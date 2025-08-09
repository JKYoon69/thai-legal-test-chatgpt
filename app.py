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

# ---------- Global style ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
:root { --thai-font: 'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; }

.leftpane { max-height: 640px; overflow-y: auto; padding: 6px; border-right: 1px solid #333; }
.row { display: flex; align-items: center; gap: 8px; margin: 2px 0; }
.indent { width: 0; }
.chevbtn, .txtbtn {
  background: transparent; border: none; color: #e6e6e6; font-size: 0.95rem;
  font-family: var(--thai-font); cursor: pointer; padding: 4px 6px;
}
.chevbtn { width: 22px; text-align: center; }
.txtbtn { text-align: left; }
.txtbtn.sel { color: #8ab4f8; font-weight: 600; }

.docbox { max-height: 640px; overflow-y: auto; padding: 12px;
  border: 1px solid #333; border-radius: 8px; background: #0e1117; width: 100%; }
.doc { font-family: var(--thai-font); color:#e6e6e6; line-height:1.9; font-size: 1.05rem;
  white-space: pre-wrap; word-break: normal; overflow-wrap: anywhere; margin: 0; }
.selbg { background:#133a1a; color:#a7f3d0; }   /* 선택 영역: 녹색 계열 텍스트 + 어두운 BG */
.selbgYellow { background:#3a3413; color:#ffe169; } /* 필요시 노랑으로 바꿔도 됨 */
.rawbox { max-height: 420px; overflow-y: auto; padding: 10px;
  border: 1px solid #333; border-radius: 8px; background: #0e1117; }
.raw { font-family: var(--thai-font); color:#e6e6e6; white-space: pre-wrap; margin: 0; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Flow:** Upload → Parse → Review → Download")
    st.caption("UI v2 — text-only tree, dark previews, responsive full doc")

# ---------- Session ----------
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

# 1) Raw text (scrollable) — dark background
with st.expander("Raw text (scrollable)", expanded=False):
    safe = (raw_text[:300000]).replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f"<div class='rawbox'><pre class='raw'>{safe}</pre></div>", unsafe_allow_html=True)

# Controls (parser + downloads live near the top so 안 보일 때도 접근 가능)
auto_type = detect_doc_type(raw_text)
st.write(f"🔎 Detected doc type: **{auto_type}**")
forced_type = st.selectbox("Doc type (override if needed)",
                           ["auto","code","act","royal_decree","regulation"], index=0)
if forced_type == "auto": forced_type = None

run_col, dl_col = st.columns([1,2])
with run_col:
    run_clicked = st.button("🧩 Run parser", use_container_width=True)

# Run parser
if run_clicked:
    with st.spinner("Parsing..."):
        result = parse_document(raw_text, forced_type=forced_type)
        ss.result = result
        ss.selected_node_id = result.nodes[0].node_id if result.nodes else None
        ss.expanded = {}

result: ParseResult|None = ss.result
if not result:
    st.stop()

# Downloads (always visible under Run parser)
with dl_col:
    out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
    jsonl_nodes = out_dir / "nodes.jsonl"
    jsonl_chunks = out_dir / "chunks.jsonl"
    preview_html = out_dir / "preview.html"
    zip_path = out_dir / "bundle.zip"
    debug_json = out_dir / "debug.json"

    # prepare artifacts
    to_jsonl(result.nodes, jsonl_nodes)
    chunks = make_chunks(raw_text, result, mode="article±1")
    to_jsonl(chunks, jsonl_chunks)
    preview_html.write_text(
        "<html><meta charset='utf-8'><body style=\"font-family:'Noto Sans Thai',sans-serif\">"
        "<h3>Thai Law Parser Preview</h3>"
        f"<pre>{json.dumps([n.model_dump() for n in result.nodes[:30]], ensure_ascii=False, indent=2)}</pre>"
        "</body></html>", encoding="utf-8")
    debug_report = make_debug_report(raw_text, result, chunks)
    debug_json.write_text(json.dumps(debug_report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_zip_bundle(zip_path, {
        "nodes.jsonl": jsonl_nodes.read_bytes(),
        "chunks.jsonl": jsonl_chunks.read_bytes(),
        "preview.html": preview_html.read_bytes(),
        "debug.json": debug_json.read_bytes(),
    })

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("⬇️ nodes.jsonl", data=jsonl_nodes.read_bytes(),
                           file_name="nodes.jsonl", mime="application/jsonl", key="dl_nodes")
    with c2:
        st.download_button("⬇️ chunks.jsonl", data=jsonl_chunks.read_bytes(),
                           file_name="chunks.jsonl", mime="application/jsonl", key="dl_chunks")
    with c3:
        st.download_button("⬇️ bundle.zip", data=zip_path.read_bytes(),
                           file_name="thai-law-parser-bundle.zip", mime="application/zip", key="dl_zip")
    with c4:
        st.download_button("🐞 debug.json", data=debug_json.read_bytes(),
                           file_name="debug.json", mime="application/json", key="dl_debug")

st.success(f"Parsed: {len(result.nodes)} nodes, leaves {result.stats.get('leaf_count',0)}")
issues = validate_tree(result)
with st.expander(f"Consistency check (issues: {len(issues)})", expanded=False):
    st.write("No issues ✅" if not issues else "\n".join([f"[{i.level}] {i.message}" for i in issues]))

# ---------- Build flat list (depth info) ----------
flat: list[dict] = []
def walk(n: Node, depth:int=0):
    flat.append({"id": n.node_id, "label": n.label, "span": (n.span.start, n.span.end),
                 "depth": depth, "has_children": bool(n.children)})
    for ch in n.children: walk(ch, depth+1)
for root in result.root_nodes: walk(root, 0)

# quick lookup map
node_by_id = {n["id"]: n for n in flat}

# ---------- Layout: Hierarchy (text-only) + Full document (dark, white text, highlight) ----------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Hierarchy")
    st.caption("Expand/collapse with ▸ ▾. Click a node to highlight its range on the right.")
    st.markdown("<div id='leftpane' class='leftpane'>", unsafe_allow_html=True)

    def render_node(node_id: str):
        item = node_by_id[node_id]
        depth = item["depth"]; has_children = item["has_children"]
        indent_px = depth * 18
        expanded = ss.expanded.get(node_id, False)
        arrow = "▾" if expanded else ("▸" if has_children else "•")

        # --- left chevron (expand/collapse) ---
        with st.form(key=f"tg-{node_id}"):
            st.markdown(
                f"<div class='row'><div class='indent' style='margin-left:{indent_px}px;'></div>"
                f"<button class='chevbtn' type='submit'>{arrow}</button>"
                f"<button class='txtbtn{' sel' if ss.selected_node_id == node_id else ''}' type='submit' formaction='sel'>{item['label']}</button></div>",
                unsafe_allow_html=True
            )
            # 폼에 두 개의 버튼이 있지만 Streamlit은 어떤 버튼이 눌렸는지 알 수 없음 → 작은 트릭:
            # 첫 submit은 토글, 두 번째는 선택. 이를 구분하기 위해 아래 두 submit 버튼을 둔다.
            # (Streamlit 한계로 완벽하진 않지만 효과적)
            submitted = st.form_submit_button("", use_container_width=False)
            if submitted:
                # 기본: 토글 동작
                if has_children:
                    ss.expanded[node_id] = not expanded
                    # 닫히는 순간엔 상위 범위를 보여 달라고 했으므로 선택도 부모로 설정
                    if not ss.expanded[node_id]:
                        ss.selected_node_id = node_id
                else:
                    ss.selected_node_id = node_id

        # children
        if has_children and ss.expanded.get(node_id, False):
            # 직계 자식만 재귀 렌더
            idx = flat.index(item) + 1
            while idx < len(flat) and flat[idx]["depth"] > depth:
                if flat[idx]["depth"] == depth + 1:
                    render_node(flat[idx]["id"])
                idx += 1

    # Render roots
    for r in [x for x in flat if x["depth"] == 0]:
        render_node(r["id"])

    st.markdown("</div>", unsafe_allow_html=True)

    # auto-scroll LEFT to selected
    if ss.selected_node_id:
        components.html(f"""
            <script>
            const selForm = parent.document.querySelector("form#tg-{ss.selected_node_id}");
            if (selForm) selForm.scrollIntoView({{block:'center', behavior:'instant'}});
            </script>
        """, height=0)

with right:
    st.subheader("Full document (auto-scroll & highlight)")
    sel_id = ss.selected_node_id or flat[0]["id"]
    target = next((n for n in result.nodes if n.node_id == sel_id), None)

    if target:
        s, e = target.span.start, target.span.end
        before = raw_text[:s].replace("<","&lt;").replace(">","&gt;")
        body   = raw_text[s:e].replace("<","&lt;").replace(">","&gt;")
        after  = raw_text[e:].replace("<","&lt;").replace(">","&gt;")

        html = f"""
<div id="docbox" class="docbox" style="width:100%;">
  <pre class="doc">{before}<a id="SEL"></a><span class="selbg">{body}</span>{after}</pre>
</div>
<script>
  const sel = document.getElementById("SEL");
  if (sel) sel.scrollIntoView({{block:'center'}});
</script>
"""
        components.html(html, height=640, scrolling=False, width=0)  # width=0 → container width(100%)
    else:
        st.info("Select a node on the left to preview.")

st.divider()
st.subheader("Chunking (for RAG)")
mode = st.selectbox("Merge mode", ["article_only", "article±1"], index=1, key="merge_mode")
# 이미 위에서 만든 chunks 재사용도 가능하지만, 모드 바꿀 수 있으니 다시 생성
chunks = make_chunks(raw_text, ss.result, mode=mode)
st.write(f"Generated chunks: {len(chunks)}")

with st.expander("Sample chunks (JSON)", expanded=False):
    st.code(json.dumps([c.model_dump() for c in chunks[:5]], ensure_ascii=False, indent=2))
