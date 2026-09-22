"""引用迁移复核：把旧版条款引用按段落映射追踪到新版。

流程（可解释、可复现，**禁止猜测**）：

1. 对**旧版**与**新版**段落分别执行现有引用解析
   （``citations.scan_citations``），两侧解析规则完全一致。
2. 仅审查**来源有新旧段落映射**的引用：取对齐记录中旧、新两侧均非空的
   映射组（一旧↔多新＝拆分、多旧↔一新＝合并、多对多＝多段改写均支持），
   把组内旧版引用与新版引用分别按 **引用类型 + 组内出现序号** 配对。
   - 同类型数量不等时，多出来的未配对引用记为 **待核验**（unpaired）；
   - 只在单侧出现来源的引用（来源段已删除/属新增内容）不在审查范围。
3. 对每一对引用，旧引用目标所在对齐记录的**全部新版段落**构成“预期集合”；
   目标本身无新旧映射时记为 **待核验**。
   - 新版引用解析为**唯一目标**，且目标覆盖的全部段落都落入预期集合
     → **已迁移**；
   - 唯一目标指向预期集合**之外的其他有效条款** → **疑似错指**
     （部分落入集合也按疑似错指，不静默收窄）；
   - 任一侧引用缺失/歧义、或目标无映射 → **待核验**。

结果在请求时即时计算、不落库：引用重扫或手动调整映射后再次请求即为最新结果。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .citations import REF_TYPE_LABEL, build_structure, scan_citations

# 迁移三态
MIGRATED = "migrated"          # 已迁移
SUSPECTED = "suspected"        # 疑似错指
REVIEW = "review"              # 待核验（含配对缺失的未配对项）

MIGRATION_STATUS_LABEL = {
    MIGRATED: "已迁移",
    SUSPECTED: "疑似错指",
    REVIEW: "待核验",
}

# 与 main.CHANGE_TYPES 保持一致的展示名（经参数注入亦可覆盖）
DEFAULT_CHANGE_TYPE_LABEL = {
    "unchanged": "未变更",
    "rewrite": "改写",
    "added": "新增",
    "deleted": "删除",
    "split": "拆分",
    "merged": "合并",
    "multi_rewrite": "多段改写",
}


@dataclass
class _Group:
    align_id: int
    change_type: str
    is_manual: bool
    old_idx: list[int]
    new_idx: list[int]
    old_cits: list[dict] = field(default_factory=list)
    new_cits: list[dict] = field(default_factory=list)


def _para_label(struct, para_idx: int) -> str:
    """新版段落的人类可读标签：第X条 / 第X条第N款 / 第N段。"""
    from .citations import _art_label, int_to_cn
    n = para_idx + 1
    art = struct.para_art.get(para_idx)
    if art is None:
        return f"新版第 {n} 段（条款结构外）"
    para_lines = [ln for ln in struct.lines if ln.para == para_idx]
    if any(ln.is_head for ln in para_lines):
        return _art_label(art)
    kuan = next((ln.kuan_no for ln in para_lines if ln.kuan_no), None)
    if kuan:
        return f"{_art_label(art)}第{int_to_cn(kuan)}款（新版第 {n} 段）"
    return f"{_art_label(art)}（新版第 {n} 段）"


def _cite_brief(c: dict, side_name: str) -> dict:
    """提取引用的展示/判定字段。"""
    return {
        "order_idx": c["order_idx"],
        "source_para_idx": c["source_para_idx"],
        "source_start": c["source_start"],
        "source_end": c["source_end"],
        "mention_text": c["mention_text"],
        "ref_type": c["ref_type"],
        "ref_type_label": REF_TYPE_LABEL.get(c["ref_type"], c["ref_type"]),
        "status": c["status"],
        "target_label": c["target_label"],
        "target_excerpt": c["target_excerpt"],
        "source_location": c["source_location"],
        "reason": c["reason"],
        "targets": c["targets"],
        "candidates": c["candidates"],
        "side": side_name,
    }


def build_migration(old_paragraphs: list[str], new_paragraphs: list[str],
                    alignments: list[dict],
                    change_type_labels: dict | None = None) -> dict:
    """计算引用迁移复核结果。

    :param old_paragraphs: 旧版段落文本（按 idx 升序）
    :param new_paragraphs: 新版段落文本
    :param alignments: 对齐记录（row_to_dict 形式，含 id/old_indices/new_indices/
                       change_type/is_manual/explanation）
    :return: ``{"n_old_mentions", "n_new_mentions", "n_groups", "counts", "items"}``；
             items 已按来源映射组与组内出现顺序排序。
    """
    ct_labels = change_type_labels or DEFAULT_CHANGE_TYPE_LABEL
    # ---- 1. 两侧分别执行现有引用解析 ----
    old_records = scan_citations(old_paragraphs, side="old")
    new_records = scan_citations(new_paragraphs, side="new")
    new_struct = build_structure(new_paragraphs)

    # 段落 idx → 归属对齐记录（段落不允许跨映射重复，映射建立时已保证）
    old_align: dict[int, dict] = {}
    groups: dict[int, _Group] = {}
    for a in alignments:
        oi, ni = list(a["old_indices"]), list(a["new_indices"])
        for i in oi:
            old_align[i] = a
        # 仅审查两侧均非空、有新旧段落对应的来源映射组
        if oi and ni:
            groups[a["id"]] = _Group(
                align_id=a["id"], change_type=a["change_type"],
                is_manual=bool(a.get("is_manual")), old_idx=oi, new_idx=ni)

    def owner_of_old_target(p: int) -> dict | None:
        return old_align.get(p)

    # ---- 2. 来源按映射组归集 ----
    old_by_para: dict[int, list[dict]] = defaultdict(list)
    for c in old_records:
        old_by_para[c["source_para_idx"]].append(c)
    new_by_para: dict[int, list[dict]] = defaultdict(list)
    for c in new_records:
        new_by_para[c["source_para_idx"]].append(c)

    for g in groups.values():
        # 组内保持段落与扫描顺序（order_idx 即文档内出现先后）
        g.old_cits = sorted((c for i in g.old_idx for c in old_by_para.get(i, [])),
                            key=lambda c: c["order_idx"])
        g.new_cits = sorted((c for i in g.new_idx for c in new_by_para.get(i, [])),
                            key=lambda c: c["order_idx"])

    # ---- 3. 组内按引用类型 + 出现序号配对 ----
    items: list[dict] = []
    for g in _groups_in_order(groups):
        group_basis = _group_basis(g, ct_labels)
        old_buckets: dict[str, list[dict]] = defaultdict(list)
        new_buckets: dict[str, list[dict]] = defaultdict(list)
        for c in g.old_cits:
            old_buckets[c["ref_type"]].append(c)
        for c in g.new_cits:
            new_buckets[c["ref_type"]].append(c)

        # 先按旧侧出现序，把每类的前 min(n_old,n_new) 项逐序配对；
        # 同类型数量不等时多出来的项作为未配对项追加，避免挤乱后续配对序号。
        old_type_order = list(dict.fromkeys(c["ref_type"] for c in g.old_cits))
        extra_types = sorted(t for t in new_buckets if t not in old_buckets)
        all_types = old_type_order + extra_types
        unpaired: list[tuple[str, int]] = []
        for rt in all_types:
            ob, nb = old_buckets[rt], new_buckets[rt]
            n_pair = min(len(ob), len(nb))
            for occ in range(n_pair):
                items.append(_evaluate(g, rt, occ, ob[occ], nb[occ], old_paragraphs,
                                       new_paragraphs, new_struct, owner_of_old_target,
                                       group_basis))
            for occ in range(n_pair, max(len(ob), len(nb))):
                unpaired.append((rt, occ))
        for rt, occ in unpaired:
            old_c = old_buckets[rt][occ] if occ < len(old_buckets[rt]) else None
            new_c = new_buckets[rt][occ] if occ < len(new_buckets[rt]) else None
            items.append(_evaluate(g, rt, occ, old_c, new_c, old_paragraphs,
                                   new_paragraphs, new_struct, owner_of_old_target,
                                   group_basis))

    counts = {MIGRATED: 0, SUSPECTED: 0, REVIEW: 0,
              "unpaired_old": 0, "unpaired_new": 0}
    for it in items:
        counts[it["status"]] += 1
        if it["pair_kind"] == "unpaired_old":
            counts["unpaired_old"] += 1
        if it["pair_kind"] == "unpaired_new":
            counts["unpaired_new"] += 1

    return {
        "n_old_mentions": len(old_records),
        "n_new_mentions": len(new_records),
        "n_groups": len(groups),
        "counts": counts,
        "items": items,
    }


def _groups_in_order(groups: dict[int, _Group]) -> list[_Group]:
    return sorted(groups.values(), key=lambda g: (min(g.old_idx), g.align_id))


def _group_basis(g: _Group, ct_labels: dict) -> str:
    old_s = "、".join(str(i + 1) for i in g.old_idx)
    new_s = "、".join(str(i + 1) for i in g.new_idx)
    manual = "（人工调整）" if g.is_manual else ""
    return (f"对齐记录 #{g.align_id}：旧版第 {old_s} 段 ↔ 新版第 {new_s} 段"
            f"，变更类型「{ct_labels.get(g.change_type, g.change_type)}」{manual}")


def _evaluate(g: _Group, ref_type: str, occ: int,
              old_c: dict | None, new_c: dict | None,
              old_paragraphs: list[str], new_paragraphs: list[str],
              new_struct, owner_of_old_target,
              group_basis: str) -> dict:
    """配对后逐条判定三态。禁止猜测：任何信息不完整一律待核验。"""
    rt_label = REF_TYPE_LABEL.get(ref_type, ref_type)
    occ_text = f"{rt_label}第 {occ + 1} 处"
    base = {
        "alignment_id": g.align_id,
        "change_type": g.change_type,
        "is_manual": g.is_manual,
        "group_old_indices": g.old_idx,
        "group_new_indices": g.new_idx,
        "group_basis": group_basis,
        "ref_type": ref_type,
        "ref_type_label": rt_label,
        "occurrence": occ + 1,
        "old_mention": None,
        "new_mention": None,
        "expected_targets": [],
        "target_basis": [],
        "status": REVIEW,
        "status_label": MIGRATION_STATUS_LABEL[REVIEW],
        "reason": "",
    }

    # ---- 同类型数量不等：未配对项 → 待核验 ----
    if old_c is None:
        base["pair_kind"] = "unpaired_new"
        base["new_mention"] = _cite_brief(new_c, "新版")
        base["reason"] = (f"{occ_text}：新版来源映射组内该类型有 "
                          f"{_bucket_len(g, ref_type, 'new')} 处、旧版仅 "
                          f"{_bucket_len(g, ref_type, 'old')} 处，"
                          "该引用无旧版同类型项可配对，需人工核验。")
        return base
    if new_c is None:
        base["pair_kind"] = "unpaired_old"
        base["old_mention"] = _cite_brief(old_c, "旧版")
        base["reason"] = (f"{occ_text}：旧版来源映射组内该类型有 "
                          f"{_bucket_len(g, ref_type, 'old')} 处、新版仅 "
                          f"{_bucket_len(g, ref_type, 'new')} 处，"
                          "该引用在新版无同类型对应项，需人工核验。")
        return base
    base["pair_kind"] = "paired"
    base["old_mention"] = _cite_brief(old_c, "旧版")
    base["new_mention"] = _cite_brief(new_c, "新版")

    # ---- 任一侧引用解析缺失/歧义 → 待核验 ----
    if old_c["status"] != "ok":
        if old_c["status"] == "missing":
            base["reason"] = f"旧版引用本身解析失败（缺失）：{old_c['reason']}，无预期目标可比，需人工核验。"
        else:
            base["reason"] = (f"旧版引用存在 {len(old_c['candidates'])} 个候选（歧义），"
                              "不静默选取，需人工核验。")
        return base
    if new_c["status"] != "ok":
        if new_c["status"] == "missing":
            base["reason"] = f"新版引用解析为缺失：{new_c['reason']}，需人工核验。"
        else:
            base["reason"] = (f"新版引用存在 {len(new_c['candidates'])} 个候选（歧义），"
                              "不静默选取，需人工核验。")
        return base

    # ---- 由旧目标所在对齐记录推导预期集合 ----
    expected: set[int] = set()
    target_basis: list[str] = []
    for t in old_c["targets"]:
        covered = t.get("para_indices") or [t["para_idx"]]
        align_ids: list[int] = []
        mapped_new: set[int] = set()
        for p in covered:
            a = owner_of_old_target(p)
            if a is None:
                base["reason"] = (
                    f"旧版引用目标「{t['label']}」覆盖旧版第 {p + 1} 段，"
                    "该段在对齐结果中无新旧段落映射（可能已删除），无法推导新版预期目标，需人工核验。")
                return base
            ni = list(a["new_indices"])
            if not ni:
                base["reason"] = (
                    f"旧版引用目标「{t['label']}」所在对齐记录 #{a['id']} "
                    "无新版对应段落（目标内容疑似删除），需人工核验。")
                return base
            if a["id"] not in align_ids:
                align_ids.append(a["id"])
            mapped_new.update(ni)
        expected.update(mapped_new)
        target_basis.append(
            f"旧目标「{t['label']}」→ 对齐记录 #{'、#'.join(str(i) for i in align_ids)}"
            f" → 新版第 {'、'.join(str(x + 1) for x in sorted(mapped_new))} 段")
    if not expected:
        base["reason"] = "旧版引用无有效目标段落，需人工核验。"
        return base

    expected_sorted = sorted(expected)
    base["expected_targets"] = [
        {"para_idx": p, "label": _para_label(new_struct, p),
         "excerpt": new_paragraphs[p]} for p in expected_sorted
    ]
    base["target_basis"] = target_basis

    # ---- 新版实际目标落点判定 ----
    actual_paras: set[int] = set()
    for t in new_c["targets"]:
        actual_paras.update(t.get("para_indices") or [t["para_idx"]])
    inside = actual_paras & expected
    outside = actual_paras - expected

    if not outside and inside:
        base["status"] = MIGRATED
        base["status_label"] = MIGRATION_STATUS_LABEL[MIGRATED]
        new_labels = "、".join(t["label"] for t in new_c["targets"])
        if actual_paras == expected:
            base["reason"] = (f"新版实际目标「{new_labels}」覆盖段落全部落入预期集合"
                              f"（新版第 {'、'.join(str(p + 1) for p in expected_sorted)} 段）。")
        else:
            base["reason"] = (
                f"新版实际目标「{new_labels}」为预期集合的子集"
                f"（实际第 {'、'.join(str(p + 1) for p in sorted(actual_paras))} 段 ⊂ "
                f"预期第 {'、'.join(str(p + 1) for p in expected_sorted)} 段），目标仍在映射范围内。")
    elif outside:
        base["status"] = SUSPECTED
        base["status_label"] = MIGRATION_STATUS_LABEL[SUSPECTED]
        out_labels = "、".join(
            _para_label(new_struct, p) for p in sorted(outside))
        in_part = (f"；其中第 {'、'.join(str(p + 1) for p in sorted(inside))} 段在预期集合内，"
                   "但目标同时指向集合外，不做静默收窄" if inside else "")
        new_labels = "、".join(t["label"] for t in new_c["targets"])
        base["reason"] = (
            f"新版实际目标「{new_labels}」指向预期集合之外的有效条款：{out_labels}"
            f"{in_part}，疑似错指，请人工核对。")
    else:
        base["reason"] = "新版实际目标未能与预期集合建立关系，需人工核验。"
    return base


def _bucket_len(g: _Group, ref_type: str, side: str) -> int:
    bucket = g.old_cits if side == "old" else g.new_cits
    return sum(1 for c in bucket if c["ref_type"] == ref_type)


def filter_items(items: list[dict], status: str | None = None,
                 ref_type: str | None = None, keyword: str | None = None) -> list[dict]:
    """按迁移三态 / 引用类型 / 关键字筛选（与前端筛选条件一致，供导出复用）。"""
    out = items
    if status:
        # 未配对项的状态同样记为 review（待核验）
        out = [it for it in out if it["status"] == status]
    if ref_type:
        out = [it for it in out if it["ref_type"] == ref_type]
    kw = (keyword or "").strip().lower()
    if kw:
        def hay(it: dict) -> str:
            parts = [it["reason"], it["group_basis"], it["ref_type_label"]]
            for side in (it.get("old_mention"), it.get("new_mention")):
                if side:
                    parts += [side["mention_text"], side["target_label"],
                              side["source_location"], side["reason"]]
            parts += [t["label"] for t in it.get("expected_targets", [])]
            return " ".join(parts).lower()
        out = [it for it in out if kw in hay(it)]
    return out
