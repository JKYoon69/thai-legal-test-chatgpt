# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any

from .schema import ParseResult, Node, Chunk
from .rules_th import normalize_text

MAX_NODE_CHARS = 10000  # 사실상 비활성

# ───────────────────────────── 레벨 랭킹 & 유틸 ───────────────────────────── #

_LEVEL_ORDER = {
    "root": 0,
    "front_matter": 1,
    "ภาค": 2,
    "ลักษณะ": 3,
    "หมวด": 4,
    "ส่วน": 5,
    # 'บทบัญญัติทั่วไป', 'บทกำหนดโทษ', 'บทเฉพาะกาล' 등은 'บท*' prefix로 취급
    "บท*": 6,
    "มาตรา": 7,
    "ข้อ": 7,
}

def _rank(label: Optional[str]) -> int:
    if not label:
        return 99
    if label.startswith("บท"):
        return _LEVEL_ORDER["บท*"]
    return _LEVEL_ORDER.get(label, 98)

def _is_article_leaf(n: Node) -> bool:
    return n.label in ("มาตรา", "ข้อ")

def _collect_article_leaves(root: Node) -> List[Node]:
    leaves: List[Node] = []
    stack = list(root.children)
    while stack:
        n = stack.pop(0)
        if _is_article_leaf(n):
            leaves.append(n)
        for c in n.children:
            stack.append(c)
    leaves.sort(key=lambda x: x.span_start)
    return leaves

def _breadcrumbs(n: Node, result: ParseResult) -> List[str]:
    crumbs = []
    cur = n
    while cur and cur.parent_id:
        crumbs.append(f"{cur.label or ''}{(' ' + cur.num) if cur.num else ''}".strip())
        cur = result.node_map.get(cur.parent_id)
        if cur and cur.label == "root":
            break
    return list(reversed(crumbs))

def _compute_union(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda x: x[0])
    merged: List[List[int]] = []
    for s, e in spans_sorted:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

def _subtract_intervals(outer: Tuple[int, int], covered: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    S, E = outer
    res: List[Tuple[int, int]] = []
    cur = S
    for s, e in covered:
        if e <= cur:
            continue
        if s > cur:
            res.append((cur, min(s, E)))
        cur = max(cur, e)
        if cur >= E:
            break
    if cur < E:
        res.append((cur, E))
    return [(s, e) for (s, e) in res if e > s]

# ───────────────────────────── 제목/연도 추출 ───────────────────────────── #

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

def _thai_to_arabic(s: str) -> str:
    return s.translate(_THAI_DIGITS)

def guess_law_name(text: str) -> Optional[str]:
    """
    제목 + 부제 + 연도(พ.ศ.)를 최대한 한 줄로 결합
    - 태국 숫자 → 아라비아 숫자 변환
    """
    norm = normalize_text(text)
    lines = [ln.strip(" \t-–—") for ln in norm.splitlines()[:80] if ln.strip()]
    base_idx = None
    for i, ln in enumerate(lines):
        if ("พระราชบัญญัติ" in ln) or ("ประมวลกฎหมาย" in ln):
            base_idx = i
            break
    if base_idx is None:
        return None

    title = lines[base_idx]
    # 다음 몇 줄에서 부제/연도 후보 찾기
    tail = ""
    for j in range(base_idx + 1, min(base_idx + 6, len(lines))):
        ln = lines[j]
        if any(k in ln for k in ("ภาค", "ลักษณะ", "หมวด", "ส่วน", "บท", "มาตรา")):
            break
        # 연도 포함 라인 우선
        if "พ.ศ." in ln or "พ. ศ." in ln or "พศ" in ln:
            tail = ln
            break
        # 부제만 있는 경우(예: "ให้ใช้ประมวลกฎหมายยาเสพติด")
        if not tail:
            tail = ln

    cand = title
    if tail:
        # 중복 슬래시 제거하고 공백으로 결합
        cand = f"{title} {tail}"

    # 숫자 정규화
    cand = _thai_to_arabic(cand)
    # 'พ.ศ.' 뒤 연도가 붙지 않은 경우 라인 전체에서 연도 재탐색
    # (간단 정규화: 'พ.ศ. ' 패턴에 아라비아 숫자 기대)
    return cand.strip()

# ───────────────────────────── 검증 ───────────────────────────── #

def validate_tree(result: ParseResult) -> List[str]:
    issues: List[str] = []

    def _check(n: Node):
        for i, c in enumerate(n.children):
            if i + 1 < len(n.children):
                nxt = n.children[i + 1]
                if c.span_end > nxt.span_start:
                    issues.append(f"Span overlap at {c.label} {c.num} -> {nxt.label} {nxt.num}")
            # child outside parent
            if c.span_start < n.span_start or c.span_end > n.span_end:
                issues.append(f"child outside parent: {c.label} {c.num} in {n.label} {n.num}")
            _check(c)
    _check(result.root)

    for n in result.all_nodes:
        if n is result.root or n.label == "front_matter":
            continue
        if n.text is None or len(n.text.strip()) < 1:
            issues.append(f"Empty node: {n.label} {n.num}")
    return issues

# ───────────────────────────── 트리 수복(Reparent & Span-Repair) ───────────────────────────── #

def repair_tree(result: ParseResult) -> Dict[str, Any]:
    """
    부모-자식 경계 불일치 최소화:
      1) bottom-up으로 부모 span을 자식 union에 맞게 확장/축소
      2) (보수적) reparent는 최소화: 부모 랭크가 자식보다 낮거나 같아도, 우선은 span 보정으로 해결
    무손실: 텍스트/스팬의 총합(coverage)에는 영향 없음.
    """
    full = result.full_text
    diag: Dict[str, Any] = {
        "adjusted_parents": 0,
        "expanded_total_chars": 0,
        "shrunk_total_chars": 0,
    }

    # depth별 정렬(깊은 노드부터 부모로)
    nodes_by_depth: Dict[int, List[Node]] = {}
    max_depth = 0
    for n in result.all_nodes:
        nodes_by_depth.setdefault(n.level, []).append(n)
        if n.level > max_depth:
            max_depth = n.level

    # 1) 부모 span 보정
    for depth in range(max_depth - 1, -1, -1):
        for p in nodes_by_depth.get(depth, []):
            if not p.children:
                # 말단: text 동기화만 보장
                if p.span_start is not None and p.span_end is not None:
                    p.text = full[p.span_start:p.span_end]
                continue
            # 자식 union 계산
            child_spans = [(c.span_start, c.span_end) for c in p.children if c.span_start is not None and c.span_end is not None]
            if not child_spans:
                continue
            new_start = min(s for s, _ in child_spans)
            new_end = max(e for _, e in child_spans)
            # 부모가 자식 union을 포함하지 못하면 보정
            if p.span_start is None or p.span_end is None:
                p.span_start, p.span_end = new_start, new_end
                p.text = full[p.span_start:p.span_end]
                diag["adjusted_parents"] += 1
                continue

            old_s, old_e = p.span_start, p.span_end
            changed = False
            if new_start < old_s:
                diag["expanded_total_chars"] += (old_s - new_start)
                old_s = new_start
                changed = True
            if new_end > old_e:
                diag["expanded_total_chars"] += (new_end - old_e)
                old_e = new_end
                changed = True
            # 부모가 자식보다 과도하게 넓은 경우(상황에 따라 축소) — 보수적으로, 자식 union보다 좌우 0~N만큼 여유가 있어도 유지 가능
            # 여기서는 안전하게 union으로 딱 맞추지 않고, 기존 부모 경계를 유지(축소는 최소화).
            # 다만 명백히 자식 union 밖의 공백만 포함할 때는 축소.
            if old_s > new_start or old_e < new_end:
                # 이미 위에서 확장했으니 축소 불가 상태
                pass
            else:
                # 부모가 자식을 충분히 포함하는데 과도하게 넓을 경우, 가장자리의 연속 공백만 있을 때만 축소
                left_pad = full[new_start: p.span_start]
                right_pad = full[p.span_end: new_end]
                # 위 계산은 old_s/old_e로 바뀌었을 수 있으므로 간단히 새로 정의
                left_pad = full[new_start: old_s]
                right_pad = full[old_e: new_end]
                # pad가 모두 공백/개행이라면 축소
                # (실제론 위에서 old_s/e를 new_*에 맞췄으므로 left_pad/right_pad는 거의 빈 문자열일 것)
                # 보수적 접근: 축소는 하지 않음(안전)
                pass

            if changed:
                p.span_start, p.span_end = old_s, old_e
                p.text = full[p.span_start:p.span_end]
                diag["adjusted_parents"] += 1

    # 2) 부모 밖으로 벗어난 자식이 남았는지 최종 점검(필요 시 최소 reparent)
    # 대부분은 부모 span 보정으로 해결됨. 예외적으로 여전히 벗어난 경우 root로 올려 임시 수복.
    root = result.root
    for p in result.all_nodes:
        for c in list(p.children):
            if c.span_start < p.span_start or c.span_end > p.span_end:
                # 최소 reparent: root로 이동 (보수적; 파괴 최소)
                try:
                    p.children.remove(c)
                except ValueError:
                    pass
                c.parent_id = root.node_id
                root.children.append(c)
                # root는 항상 전체를 커버하므로 불일치 해소
                diag["adjusted_parents"] += 1

    # node_map 재구성(안전)
    result.node_map = {n.node_id: n for n in result.all_nodes}
    return diag

# ───────────────────────────── 조문 시리즈/가중치 ───────────────────────────── #

def _article_series_index(leaves: List[Node]) -> Dict[str, int]:
    def main_int(num: Optional[str]) -> int:
        if not num:
            return 10**9
        try:
            return int(str(num).split("/")[0])
        except Exception:
            return 10**9

    series = 1
    prev = -1
    mapping: Dict[str, int] = {}
    for lf in leaves:
        cur = main_int(lf.num)
        if prev != -1 and cur < prev:
            series += 1
        mapping[lf.node_id] = series
        prev = cur
    return mapping

def _find_path_at_offset(result: ParseResult, offset: int) -> List[str]:
    path: List[str] = []
    def descend(n: Node):
        nonlocal path
        for c in n.children:
            if c.span_start <= offset < c.span_end:
                label = f"{c.label or ''}{(' ' + c.num) if c.num else ''}".strip()
                if label:
                    path.append(label)
                descend(c)
                break
    descend(result.root)
    return path

def _retrieval_weight_for(type_: str) -> float:
    return {
        "article": 1.0,
        "appendix": 0.8,
        "headnote": 0.5,
        "front_matter": 0.3,
        "orphan_gap": 0.2,
    }.get(type_, 0.5)

# ───────────────────────────── 청크 생성 ───────────────────────────── #

def make_chunks(
    result: ParseResult,
    mode: str = "article_only",
    source_file: Optional[str] = None,
    law_name: Optional[str] = None,
    *,
    include_front_matter: bool = True,
    include_headnotes: bool = True,
    include_gap_fallback: bool = True,
    allowed_headnote_levels: List[str] = ("ภาค","ลักษณะ","หมวด","ส่วน","บท"),
    min_headnote_len: int = 24,
    min_gap_len: int = 24,
    strict_lossless: bool = False,
    # ⬇️ A단계: 롱 조문 보조분할 옵션(기본 OFF)
    split_long_articles: bool = False,
    split_threshold_chars: int = 1800,
) -> Tuple[List[Chunk], Dict[str, Any]]:
    """
    Strict 모드:
      - headnote/gap 길이 필터 비활성
      - 모든 정리 후 post-fill로 남은 간극을 orphan_gap으로 보강(공백-only 포함)
    + 회계(diag): 제거 사유 세분화
    + 롱 조문 보조분할(옵션)
    """
    text = result.full_text
    total_len = len(text)
    chunks: List[Chunk] = []
    diag: Dict[str, Any] = {
        "headnote_candidates": 0,
        "headnote_after_filter": 0,
        "final_headnote_count": 0,
        "removals": {
            "removed_short": 0,
            "removed_same_span_dedupe": 0,
            "removed_overlap_clip_to_zero": 0,
            "removed_empty_text": 0,
            "removed_invalid_span": 0,
        },
        "strict_post_fill": {"enabled": bool(strict_lossless), "filled_gaps": 0, "total_chars": 0, "spans": []},
        "allowed_headnote_levels_effective": list(allowed_headnote_levels) + ["บท* (prefix)"],
    }

    # 1) article leaves
    leaves = _collect_article_leaves(result.root)
    series_map = _article_series_index(leaves)

    def _split_paragraph_smart(s: int, e: int) -> List[Tuple[int, int]]:
        if not split_long_articles or (e - s) <= split_threshold_chars:
            return [(s, e)]
        spans: List[Tuple[int, int]] = []
        p = s
        while p < e:
            q = min(p + split_threshold_chars, e)
            # 문단 경계(빈 줄) 기준으로 자르되, 너무 짧으면 그냥 강제 컷
            cut = text.rfind("\n\n", p + int(0.6 * split_threshold_chars), q)
            if cut == -1 or cut <= p + 200:
                cut = q
            spans.append((p, cut))
            p = cut
        return spans

    article_index = 0
    for leaf in leaves:
        article_index += 1
        section_label = f"{leaf.label} {leaf.num}".strip()
        crumbs = _breadcrumbs(leaf, result)
        section_uid = "/".join(crumbs) if crumbs else section_label

        spans = _split_paragraph_smart(leaf.span_start, leaf.span_end)
        for k, (s, e) in enumerate(spans, 1):
            label = section_label + (f" (part {k})" if len(spans) > 1 else "")
            meta = {
                "type": "article",
                "mode": mode,
                "doc_type": result.doc_type,
                "law_name": law_name or "",
                "section_label": label,
                "section_uid": section_uid,
                "source_file": source_file or "",
                "article_index_global": str(article_index),
                "series": str(series_map.get(leaf.node_id, 1)),
                "retrieval_weight": str(_retrieval_weight_for("article")),
                "part": str(k) if len(spans) > 1 else "1",
            }
            chunks.append(
                Chunk(
                    text=text[s:e],
                    span_start=s,
                    span_end=e,
                    node_ids=[leaf.node_id],
                    breadcrumbs=crumbs,
                    meta=meta,
                )
            )

    # 2) front_matter
    if include_front_matter:
        for n in result.root.children:
            if n.label == "front_matter" and (n.span_end - n.span_start) > 0:
                frag = text[n.span_start:n.span_end]
                if strict_lossless or frag.strip():
                    crumbs = ["front_matter"]
                    meta = {
                        "type": "front_matter",
                        "mode": mode,
                        "doc_type": result.doc_type,
                        "law_name": law_name or "",
                        "section_label": "front_matter",
                        "section_uid": "front_matter",
                        "source_file": source_file or "",
                        "retrieval_weight": str(_retrieval_weight_for("front_matter")),
                    }
                    chunks.append(
                        Chunk(
                            text=frag,
                            span_start=n.span_start,
                            span_end=n.span_end,
                            node_ids=[n.node_id],
                            breadcrumbs=crumbs,
                            meta=meta,
                        )
                    )
                break

    # 3) headnotes (레벨+이명 허용 / Strict면 길이 필터 해제)
    if include_headnotes:
        allowed = set(allowed_headnote_levels)

        def is_allowed_label(lbl: Optional[str]) -> bool:
            if not lbl:
                return False
            return (lbl in allowed) or lbl.startswith("บท")  # 이명 허용

        def walk(parent: Node):
            nonlocal diag
            if is_allowed_label(parent.label):
                child_spans = [(c.span_start, c.span_end) for c in parent.children]
                covered = _compute_union(child_spans)
                leftovers = _subtract_intervals((parent.span_start, parent.span_end), covered)
                for (s, e) in leftovers:
                    frag = text[s:e]
                    diag["headnote_candidates"] += 1
                    keep = (e - s) > 0 if strict_lossless else (len(frag.strip()) >= min_headnote_len)
                    if keep:
                        crumbs = _breadcrumbs(parent, result)
                        section_label = f"{parent.label} {parent.num}".strip() if parent.num else parent.label or "headnote"
                        section_uid = ("/".join(crumbs) if crumbs else section_label) + " — headnote"
                        meta = {
                            "type": "headnote",
                            "mode": mode,
                            "doc_type": result.doc_type,
                            "law_name": law_name or "",
                            "section_label": f"{section_label} — headnote",
                            "section_uid": section_uid,
                            "source_file": source_file or "",
                            "retrieval_weight": str(_retrieval_weight_for("headnote")),
                        }
                        chunks.append(
                            Chunk(
                                text=frag,
                                span_start=s,
                                span_end=e,
                                node_ids=[parent.node_id],
                                breadcrumbs=crumbs,
                                meta=meta,
                            )
                        )
                        diag["headnote_after_filter"] += 1
                    else:
                        diag["removals"]["removed_short"] += 1
            for c in parent.children:
                walk(c)
        walk(result.root)

    # 4) 1차 gap-sweeper
    if include_gap_fallback:
        ivs = _compute_union([(c.span_start, c.span_end) for c in chunks])
        gaps: List[Tuple[int, int]] = []
        prev = 0
        for s, e in ivs:
            if s > prev:
                gaps.append((prev, s))
            prev = e
        if prev < total_len:
            gaps.append((prev, total_len))

        for idx, (s, e) in enumerate(gaps, 1):
            if e <= s:
                diag["removals"]["removed_invalid_span"] += 1
                continue
            frag = text[s:e]
            keep = (e - s) > 0 if strict_lossless else (len(frag.strip()) >= min_gap_len)
            if keep:
                crumbs = _find_path_at_offset(result, s)
                meta = {
                    "type": "orphan_gap",
                    "mode": mode,
                    "doc_type": result.doc_type,
                    "law_name": law_name or "",
                    "section_label": f"orphan_gap #{idx}",
                    "section_uid": f"gap[{s}:{e}]",
                    "source_file": source_file or "",
                    "retrieval_weight": str(_retrieval_weight_for("orphan_gap")),
                }
                chunks.append(
                    Chunk(
                        text=frag,
                        span_start=s,
                        span_end=e,
                        node_ids=[],
                        breadcrumbs=crumbs,
                        meta=meta,
                    )
                )
            else:
                diag["removals"]["removed_short"] += 1  # gap이 너무 짧아 버림(Strict OFF)

    # ── 정리: 정렬 → 동기화 → 동일구간 dedupe → 연속 겹침 클립 ── #
    chunks.sort(key=lambda c: (c.span_start, c.span_end))

    cleaned: List[Chunk] = []
    for c in chunks:
        s, e = c.span_start, c.span_end
        if s is None or e is None or e <= s:
            diag["removals"]["removed_invalid_span"] += 1
            continue
        c.text = text[s:e]  # 원문 재동기화
        if not strict_lossless and c.text.strip() == "":
            diag["removals"]["removed_empty_text"] += 1
            continue
        cleaned.append(c)
    chunks = cleaned

    # 동일 (span_start, span_end) 중복 제거
    seen_span = set()
    dedup1: List[Chunk] = []
    for c in chunks:
        key = (c.span_start, c.span_end)
        if key in seen_span:
            diag["removals"]["removed_same_span_dedupe"] += 1
            continue
        seen_span.add(key)
        dedup1.append(c)
    chunks = dedup1

    # 연속 겹침 → 뒤 청크 전방 클립
    final: List[Chunk] = []
    last_end = -1
    for c in chunks:
        s, e = c.span_start, c.span_end
        if last_end > -1 and s < last_end:
            s = last_end  # 전방 클립
            if e - s <= 0:
                diag["removals"]["removed_overlap_clip_to_zero"] += 1
                continue

        # Strict일 때는 headnote/gap/front_matter 길이 필터 완전 비활성화
        if not strict_lossless:
            if c.meta.get("type") in ("headnote", "orphan_gap", "front_matter"):
                thr = min_headnote_len if c.meta.get("type") == "headnote" else min_gap_len
                if (e - s) < thr:
                    diag["removals"]["removed_short"] += 1
                    continue

        c.span_start, c.span_end = s, e
        c.text = text[s:e]
        final.append(c)
        last_end = e

    # headnote 최종 개수 기록
    diag["final_headnote_count"] = sum(1 for c in final if c.meta.get("type") == "headnote")

    # ── Strict 전용 post-fill: 마지막까지 남은 간극 모두 메움 ── #
    if strict_lossless:
        ivs2 = _compute_union([(c.span_start, c.span_end) for c in final])
        post_gaps: List[Tuple[int, int]] = []
        prev = 0
        for s, e in ivs2:
            if s > prev:
                post_gaps.append((prev, s))
            prev = e
        if prev < total_len:
            post_gaps.append((prev, total_len))

        for (s, e) in post_gaps:
            if e <= s:
                diag["removals"]["removed_invalid_span"] += 1
                continue
            frag = text[s:e]   # 공백-only라도 포함
            crumbs = _find_path_at_offset(result, s)
            meta = {
                "type": "orphan_gap",
                "mode": mode,
                "doc_type": result.doc_type,
                "law_name": law_name or "",
                "section_label": f"strict_fill_gap[{s}:{e}]",
                "section_uid": f"strict_gap[{s}:{e}]",
                "source_file": source_file or "",
                "retrieval_weight": str(_retrieval_weight_for("orphan_gap")),
                "strict_fill": "1",
            }
            final.append(
                Chunk(
                    text=frag,
                    span_start=s,
                    span_end=e,
                    node_ids=[],
                    breadcrumbs=crumbs,
                    meta=meta,
                )
            )
            diag["strict_post_fill"]["filled_gaps"] += 1
            diag["strict_post_fill"]["total_chars"] += (e - s)
            if len(diag["strict_post_fill"]["spans"]) < 5:
                diag["strict_post_fill"]["spans"].append([s, e])

        final.sort(key=lambda c: (c.span_start, c.span_end))

    return final, diag
