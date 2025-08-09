# app.py — Thai Law Parser (Cloud-safe UI, Virtual hierarchy, wide doc pane)
import hashlib, json
from pathlib import Path
import datetime as _dt
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container

# ---- local modules (unchanged) ----
from parser_core.parser import detect_doc_type, parse_document
from parser_core.postprocess import validate_tree, make_chunks
from parser_core.schema import ParseResult, Node
from exporters.writers import to_jsonl, make_zip_bundle, make_debug_report

BUILD_ID = "ui-virtual-wide-v1 " + _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
st.set_page_config(page_title="Thai Law Parser — Test", layout="wide")
st.title("📜 Thai Law Parser — Test")
st.caption(f"Build: {BUILD_ID}")

# ------------------- GLOBAL CSS -------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
:root { --thai-font: 'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; }

/* 페이지 전체 폭을 뷰포트 기준으로 확장 */
.main .block-container { max-width: min(98vw, 1800px); padding-left: 1.5rem; padding-right: 1.5rem; }

/* Raw text */
.rawbox { max-height: 420px; overflow-y:auto; padding:10px; border:1px solid #333; border-radius:8px; background:#0e1117; }
.raw    { font-family: var(--thai-font); color:#e6e6e6; white-space:pre-wrap; margin:0; }

/* 왼쪽 트리 */
.hi-tree { max-height: 680px; overflow-y:auto; padding:6px 4px; border-right:1px solid #333; }
.hi-row  { display:flex; align-items:center; gap:10px; margin:8px 0; }

/* 트리 버튼을 텍스트처럼 (스코프 한정) */
#tree-scope .stButton > button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  padding: 2px 4px !important; color: #e6e6e6 !important; border-radius: 0 !important;
  font-family: var(--thai-font);
}
#tree-scope .stButton > button:hover { text-decoration: underline; }

/* 오른쪽 문서 */
.docwrap { width:100%; }
.docbox  { max-height: 840px; overflow-y:auto; padding:18px; border:1px solid #333; border-radius:10px;
           background:#0e1117; width:100%; }
.doc     { font-family:'Noto Sans Thai', Tahoma, 'Segoe UI', Arial, sans-serif; color:#e6e6e6;
           line-height:1.95; font-size:1.06rem; white-space:pre-wrap; overflow-wrap:anywhere; margin:0; }
.hlY { background:#3a3413; color:#ffe169; }
.hlG { background:#133a1a; color:#a7f3d0; }

/* Streamlit의 iframe 폭을 강제로 100% (Cloud용) */
div[data-testid="stIFrame"] iframe { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.markdown("**Utilities**")
    if st.button("♻️ Clear data/resource cache"):
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass
        st.success("Cleared cache. Rerunning…")
        st.rerun()

# ------------------- SESSION -------------------
ss = st.session_state
ss.setdefault("raw_text", None)
ss.setdefault("upload_sig", None)
ss.setdefault("result", None)
ss.setdefault("selected_node_id", None)
ss.setdefault("expanded", {})  # node_id -> bool

def _file_sig(file) -> str:
    data = file.getbuffer(); import hashlib
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

# ------------------- UPLOAD -------------------
uploaded = st.file_uploader("Upload Thai legal text (.txt)", type=["txt"])
if uploaded is not None:
    sig = _file_sig(uploaded)
    if ss.upload_sig != sig:
        ss.upload_sig = sig
        ss.raw_text = uploaded.read().decode("utf-8", errors="ignore")
        ss.result = None; ss.selected_node_id = None; ss.expanded = {}

if not ss.raw_text:
    st.info("Upload a .txt file to begin."); st.stop()
raw_text = ss.raw_text

# ------------------- RAW TEXT -------------------
with st.expander("Raw text (scrollable)", expanded=False):
    safe = raw_text[:300000].replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"<div class='rawbox'><pre class='raw'>{safe}</pre></div>", unsafe_allow_html=True)

# ------------------- PARSE -------------------
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
if not result: st.stop()

# ------------------- DOWNLOADS -------------------
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
    st.write("No issues ✅" if not issues else "\n".join(f"[{i.level}] {i.message}" for i in issues))

# ------------------- FLAT (from parsed) -------------------
flat, parents = [], {}
def walk(n: Node, depth:int=0, parent:str|None=None):
    flat.append({"id": n.node_id, "label": n.label, "span": (n.span.start, n.span.end),
                 "depth": depth, "has_children": bool(n.children)})
    parents[n.node_id] = parent
    for ch in n.children: walk(ch, depth+1, n.node_id)
for r in result.root_nodes: walk(r, 0, None)
by_id = {x["id"]: x for x in flat}

# ------------------- VIRTUAL HIERARCHY (header-based) -------------------
import re
def header_rank(lbl: str) -> int:
    # lower is higher-level
    t = lbl.strip()
    if t in ("prologue","front_matter"): return 0
    if re.match(r"^ภาค[ \u00A0]", t):      return 1
    if re.match(r"^ลักษณะ[ \u00A0]", t):  return 2
    if re.match(r"^หมวด[ \u00A0]", t):     return 3
    if re.match(r"^มาตรา[ \u00A0]", t):    return 4
    # catch “บทเฉพาะกาล/บทบัญญัติทั่วไป” 등은 2~3단으로 처리
    if "บทเฉพาะกาล" in t or "บทบัญญัติทั่วไป" in t: return 2
    return 5  # fallback (article 이하)

def build_virtual_tree():
    items = sorted([{**x} for x in flat], key=lambda k: k["span"][0])  # by doc order
    v_par = {}
    v_children = {x["id"]: [] for x in items}
    stack = []
    for it in items:
        lvl = header_rank(it["label"])
        # pop until parent level < current level
        while stack and header_rank(stack[-1]["label"]) >= lvl:
            stack.pop()
        parent = stack[-1]["id"] if stack else None
        v_par[it["id"]] = parent
        it["depth"] = (header_rank(it["label"]))  # depth = level
        if parent: v_children[parent].append(it["id"])
        stack.append(it)
    # add has_children flag
    for it in items:
        it["has_children"] = len(v_children[it["id"]]) > 0
    return items, v_par, v_children

v_flat, v_parents, v_children = build_virtual_tree()

# ------------------- UI: HIERARCHY MODE -------------------
mode = st.radio("Display mode", ["Virtual (header-based)", "Parsed (from engine)"], horizontal=True, index=0)
use_virtual = (mode.startswith("Virtual"))

# 데이터 뷰 포인터 선택
cur_flat      = v_flat if use_virtual else flat
cur_parents   = v_parents if use_virtual else parents
cur_by_id     = {x["id"]: x for x in cur_flat}

# ------------------- LAYOUT -------------------
left, right = st.columns([1, 6], gap="large")  # 오른쪽 넓게

with left:
    st.subheader("Hierarchy ↪")
    st.caption("Expand with ▸, collapse with ▾. Click a label to highlight its range on the right.")

    with stylable_container(key="tree-scope", css_styles=""):  # 스타일은 상단 CSS에서 #tree-scope로 지정
        def render_node(node_id: str):
            item = cur_by_id[node_id]
            depth = item["depth"]; has_children = item["has_children"]
            expanded = ss.expanded.get(node_id, False)
            arrow = "▾" if expanded else ("▸" if has_children else "•")

            c1, c2 = st.columns([0.12, 0.88])
            with c1:
                if has_children and st.button(arrow, key=f"tg-{mode}-{node_id}"):
                    ss.expanded[node_id] = not expanded
                elif not has_children:
                    st.write(" ")
            with c2:
                indent = " " * depth  # EM space
                if st.button(f"{indent}{item['label']}", key=f"sel-{mode}-{node_id}"):
                    ss.selected_node_id = node_id

            if has_children and ss.expanded.get(node_id, False):
                # children
                children_ids = (v_children[node_id] if use_virtual
                                else [ch["id"] for ch in cur_flat if cur_parents.get(ch["id"]) == node_id])
                for cid in children_ids:
                    render_node(cid)

        # roots
        roots = [x for x in cur_flat if cur_parents.get(x["id"]) is None]
        for r in roots:
            render_node(r["id"])

def compute_target(selected_id: str) -> str:
    # 부모가 접혀있으면 부모 범위를, 아니면 선택 노드 범위를 표시
    target = selected_id
    cur = selected_id
    while True:
        p = cur_parents.get(cur)
        if p is None: break
        if (cur_by_id[p]["has_children"] and not ss.expanded.get(p, False)):
            target = p; break
        cur = p
    return target

with right:
    st.subheader("Full document (auto-scroll & highlight)")
    color = st.radio("Highlight color", ["Yellow","Green"], horizontal=True, index=1)

    # 선택
    default_id = (cur_flat[0]["id"] if cur_flat else None)
    sel = ss.selected_node_id or default_id
    if not sel: st.stop()
    target_id = compute_target(sel)
    node = next((n for n in result.nodes if n.node_id == target_id), None)

    if node:
        s, e  = node.span.start, node.span.end
        before = raw_text[:s].replace("<","&lt;").replace(">","&gt;")
        body   = raw_text[s:e].replace("<","&lt;").replace(">","&gt;")
        after  = raw_text[e:].replace("<","&lt;").replace(">","&gt;")
        hl_cls = "hlG" if color == "Green" else "hlY"

        # ---- 방법 A: iframe (오토스크롤 포함, 폭은 CSS로 100%) ----
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8" />
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;600&display=swap');
body {{ margin:0; background:#0e1117; }}
.docwrap {{ width:100%; }}
.docbox  {{ max-height:840px; overflow-y:auto; padding:18px; border:1px solid #333; border-radius:10px; background:#0e1117; width:100%; }}
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
<script> document.getElementById("SEL")?.scrollIntoView({{block:'center'}}); </script>
</body></html>"""
        components.html(html, height=860, width=0, scrolling=False)  # width=0 → column 100%

        # ---- 방법 B: 마크다운 직접 렌더 (오토스크롤 X, 필요시 교체)
        # st.markdown(f"<div class='docbox'><pre class='doc'>{before}<span class='{hl_cls}'>{body}</span>{after}</pre></div>",
        #             unsafe_allow_html=True)
    else:
        st.info("Select a node on the left to preview.")

# ------------------- CHUNKING -------------------
st.divider()
st.subheader("Chunking (for RAG)")
mode2 = st.selectbox("Merge mode", ["article_only","article±1"], index=1)
chunks2 = make_chunks(raw_text, ss.result, mode=mode2)
st.write(f"Generated chunks: {len(chunks2)}")
with st.expander("Sample chunks (JSON)", expanded=False):
    st.code(json.dumps([c.model_dump() for c in chunks2[:5]], ensure_ascii=False, indent=2))
