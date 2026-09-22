"""引用迁移核验：旧版引用是否随段落映射正确迁移到新版。

设计原则（确定性、可解释、**禁止猜测**）：

1. 对旧版与新版文本**分别执行现有引用解析**（``citations.scan_citations``），
   不跨版本解析，两侧各自得到 正常(ok) / 缺失(missing) / 歧义(ambiguous)。
2. **仅审查有新旧段落映射的来源**：只处理 ``old_indices`` 与 ``new_indices``
   均非空的对齐记录；纯新增 / 纯删除映射组内的引用不在审查范围。
3. 组内按**引用类型**分桶，同类型引用按出现位置（段落序号 + 字符偏移）赋予
   出现序号，旧第 k 处与新第 k 处配对；同类型数量不等时，多出的未配对项
   一律标为 **待核验**。
4. 配对项的**预期集合**：旧版引用的每个目标段落，查其所在的对齐记录，
   该记录的全部新版段落并入预期集合（条目标聚合整条归属段落）。
5. 三态判定：
   - 新版实际目标段落**全部落入预期集合** → ``migrated``（已迁移）；
   - 新版解析正常但实际目标指向**预期集合外的其他有效条款** → ``suspected``
     （疑似错指）；
   - 任一侧引用缺失/歧义、目标无段落映射、或配对缺失 → ``review``
     （待核验），绝不猜测。
"""
from __future__ import annotations

from .citations import REF_TYPE_LABEL, _art_label, build_structure, scan_citations
from .database import row_to_dict

MIGRATION_STATUS_LABEL = {
    "migrated": "已迁移",
    "suspected": "疑似错指",
    "review": "待核验",
}

SIDE_STATUS_LABEL = {"ok": "正常", "missing": "缺失", "ambiguous": "歧义"}

# 单条引用记录需要带入响应的字段
_CIT_FIELDS = (
    "order_idx", "source_para_idx", "source_start", "source_end",
    "mention_text", "ref_type", "article_no", "kuan_no", "xiang_no",
    "relative", "status", "target_label", "target_excerpt",
    "source_location", "reason", "targets", "candidates",
)


def _scan_side(paragraphs: list[str], side: str) -> list[dict]:
    """对一侧文本执行现有引用解析，并把措辞前缀改为对应版本。"""
    records = scan_citations(paragraphs)
    if side == "旧":
        # resolve() 生成的说明固定写作“新版”，旧版侧结果需改写以免误导
        for r in records:
            r["source_location"] = r["source_location"].replace("新版", "旧版", 1)
            r["reason"] = r["reason"].replace("新版", "旧版")
    return records


def _cite_brief(r: dict) -> dict:
    return {k: r[k] for k in _CIT_FIELDS}


def _type_occurrence(records: list[dict], para_set: set[int]) -> dict[str, list[dict]]:
    """组内按引用类型分桶，桶内按出现位置（段落、偏移、扫描序）排序。"""
    buckets: dict[str, list[dict]] = {}
    for r in records:
        if r["source_para_idx"] in para_set:
            buckets.setdefault(r["ref_type"], []).append(r)
    for lst in buckets.values():
        lst.sort(key=lambda r: (r["source_para_idx"], r["source_start"], r["order_idx"]))
    return buckets


def _article_target_paras(target: dict, struct) -> list[int]:
    """条级引用（无款/项号）的目标覆盖整条归属段落；款/项引用只取其所在段。"""
    p = target["para_idx"]
    if target.get("kuan_no") is None and target.get("xiang_no") is None:
        art = struct.para_art.get(p)
        if art is not None:
            return struct.art_paras.get(art, []) or [p]
    return [p]


def analyze_migration(old_paragraphs: list[str], new_paragraphs: list[str],
                      alignments: list[dict]) -> dict:
    """计算引用迁移核验结果。

    ``alignments`` 为对齐记录 dict（含 ``old_indices/new_indices`` 列表）。
    """
    old_cits = _scan_side(old_paragraphs, "旧")
    new_cits = _scan_side(new_paragraphs, "新")
    old_struct = build_structure(old_paragraphs)
    new_struct = build_structure(new_paragraphs)

    # 段落 → 其所在对齐记录（同一段落不会同时出现在两条映射中）
    old_para_align: dict[int, dict] = {}
    new_para_align: dict[int, dict] = {}
    groups: list[dict] = []
    for a in alignments:
        if not a["old_indices"] or not a["new_indices"]:
            continue  # 纯新增/纯删除：无新旧映射的来源不审查
        groups.append(a)
        for i in a["old_indices"]:
            old_para_align.setdefault(i, a)
        for i in a["new_indices"]:
            new_para_align.setdefault(i, a)

    def new_para_label(np: int) -> str:
        art = new_struct.para_art.get(np)
        return f"新版第 {np + 1} 段" + (f"（{_art_label(art)}）" if art is not None else "")

    def expected_targets(old_rec: dict):
        """旧引用目标 → 预期新版段落集合。

        返回 (expected_entries, unmapped_old_paras, via_align_ids)。
        条级目标按整条归属段落逐段查映射；款/项目标只查其所在段。
        """
        expected: dict[int, dict] = {}
        unmapped: list[int] = []
        via_ids: list[int] = []
        for t in old_rec["targets"]:
            for op in _article_target_paras(t, old_struct):
                al = old_para_align.get(op)
                if al is None:
                    if op not in unmapped:
                        unmapped.append(op)
                    continue
                if al["id"] not in via_ids:
                    via_ids.append(al["id"])
                for np in al["new_indices"]:
                    expected.setdefault(np, {
                        "new_para_idx": np,
                        "label": new_para_label(np),
                        "excerpt": new_paragraphs[np],
                        "via_alignment_id": al["id"],
                    })
        return list(expected.values()), unmapped, via_ids

    def group_basis(a: dict, ref_type: str, i: int, n_old: int, n_new: int) -> str:
        type_label = REF_TYPE_LABEL.get(ref_type, ref_type)
        old_ps = "、".join(f"旧版第{x + 1}段" for x in a["old_indices"])
        new_ps = "、".join(f"新版第{x + 1}段" for x in a["new_indices"])
        manual = "，人工调整" if a.get("is_manual") else ""
        return (
            f"对齐记录 #{a['id']}（组内来源映射：{old_ps} ↔ {new_ps}{manual}）；"
            f"组内「{type_label}」旧版 {n_old} 处、新版 {n_new} 处，"
            f"按出现序号第 {i + 1} 处配对"
        )

    items: list[dict] = []

    def make_item(*, status, reason, basis, alignment, ref_type, ordinal,
                  n_old, n_new, old_rec, new_rec, expected,
                  unmapped_old=None, via_ids=None):
        unmapped_old = unmapped_old or []
        via_ids = via_ids or []
        notes = []
        if unmapped_old:
            notes.append("旧版目标含未参与任何映射的段落（"
                         + "、".join(f"旧版第{p + 1}段" for p in unmapped_old)
                         + "），该部分无法核验迁移去向")
        if expected:
            notes.append("预期集合 = 旧目标所在对齐记录（#"
                         + "、".join(str(x) for x in via_ids)
                         + "）的全部新版段落："
                         + "、".join(e["label"] for e in expected))
        full_basis = basis
        if notes:
            full_basis += "；" + "；".join(notes)
        items.append({
            "status": status,
            "reason": reason,
            "mapping_basis": full_basis,
            "alignment_id": alignment["id"],
            "ref_type": ref_type,
            "pair_ordinal": ordinal + 1 if ordinal is not None else None,
            "pair_old_count": n_old,
            "pair_new_count": n_new,
            "old": _cite_brief(old_rec) if old_rec else None,
            "new": _cite_brief(new_rec) if new_rec else None,
            "expected_targets": expected or [],
        })

    for a in groups:
        old_set, new_set = set(a["old_indices"]), set(a["new_indices"])
        old_buckets = _type_occurrence(old_cits, old_set)
        new_buckets = _type_occurrence(new_cits, new_set)
        for ref_type in sorted(set(old_buckets) | set(new_buckets)):
            ol = old_buckets.get(ref_type, [])
            nl = new_buckets.get(ref_type, [])
            for i in range(max(len(ol), len(nl))):
                o = ol[i] if i < len(ol) else None
                n = nl[i] if i < len(nl) else None
                basis = group_basis(a, ref_type, min(i, len(ol) - 1) if ol else i,
                                    len(ol), len(nl))
                # 同类型数量不等：未配对项标为待核验（旧引用侧仍给出预期目标供参考）
                if o is None or n is None:
                    expected, unmapped_old, via_ids = ([], [], [])
                    if o is not None and o["status"] == "ok":
                        expected, unmapped_old, via_ids = expected_targets(o)
                    side_note = "旧版" if n is None else "新版"
                    reason = (
                        f"组内同类型引用数量不等（旧版 {len(ol)} 处、新版 {len(nl)} 处），"
                        f"第 {i + 1} 处在{side_note}无配对引用，禁止猜测，列为待核验"
                    )
                    make_item(status="review", reason=reason, basis=basis,
                              alignment=a, ref_type=ref_type, ordinal=i,
                              n_old=len(ol), n_new=len(nl),
                              old_rec=o, new_rec=n, expected=expected,
                              unmapped_old=unmapped_old, via_ids=via_ids)
                    continue

                # 配对项：先算预期集合，再逐侧检查
                expected, unmapped_old, via_ids = ([], [], [])
                if o["status"] != "ok":
                    reason = (f"旧版引用解析为「{SIDE_STATUS_LABEL[o['status']]}」，"
                              f"无法确定迁移前目标：{o['reason']}")
                else:
                    expected, unmapped_old, via_ids = expected_targets(o)
                    if not expected:
                        reason = ("旧版引用目标段落未出现在任何新旧映射中（目标无映射），"
                                  "无法建立预期目标集合，禁止猜测")
                    elif n["status"] != "ok":
                        reason = (f"新版引用解析为「{SIDE_STATUS_LABEL[n['status']]}」，"
                                  f"无法取得实际目标进行比对：{n['reason']}")
                    else:
                        actual_paras = sorted({t["para_idx"] for t in n["targets"]})
                        unmapped_new = [p for p in actual_paras
                                        if p not in new_para_align]
                        expected_set = {e["new_para_idx"] for e in expected}
                        if unmapped_new:
                            reason = ("新版实际目标段落（"
                                      + "、".join(f"新版第{p + 1}段" for p in unmapped_new)
                                      + "）未参与任何新旧映射（目标无映射），禁止猜测")
                        elif set(actual_paras) <= expected_set:
                            labels = "、".join(t["label"] for t in n["targets"])
                            reason = (f"新版实际目标（{labels}）全部落入预期集合 "
                                      + "、".join(e["label"] for e in expected)
                                      + "，引用已随段落映射迁移")
                            make_item(status="migrated", reason=reason, basis=basis,
                                      alignment=a, ref_type=ref_type, ordinal=i,
                                      n_old=len(ol), n_new=len(nl),
                                      old_rec=o, new_rec=n, expected=expected,
                                      unmapped_old=unmapped_old, via_ids=via_ids)
                            continue
                        else:
                            outside = sorted(set(actual_paras) - expected_set)
                            out_desc = []
                            for p in outside:
                                tgt = next((t for t in n["targets"]
                                            if t["para_idx"] == p), None)
                                lab = tgt["label"] if tgt else new_para_label(p)
                                al = new_para_align.get(p)
                                if al is not None:
                                    old_ps = "、".join(f"旧版第{x + 1}段"
                                                      for x in al["old_indices"])
                                    lab += f"（所在映射 #{al['id']}，对应{old_ps}）"
                                out_desc.append(lab)
                            reason = (
                                "新版实际目标指向预期集合外的其他有效条款："
                                + "、".join(out_desc)
                                + "；预期集合为 "
                                + "、".join(e["label"] for e in expected)
                            )
                            make_item(status="suspected", reason=reason, basis=basis,
                                      alignment=a, ref_type=ref_type, ordinal=i,
                                      n_old=len(ol), n_new=len(nl),
                                      old_rec=o, new_rec=n, expected=expected,
                                      unmapped_old=unmapped_old, via_ids=via_ids)
                            continue
                # 落到这里均为待核验
                make_item(status="review", reason=reason, basis=basis,
                          alignment=a, ref_type=ref_type, ordinal=i,
                          n_old=len(ol), n_new=len(nl),
                          old_rec=o, new_rec=n, expected=expected,
                          unmapped_old=unmapped_old, via_ids=via_ids)

    counts = {"migrated": 0, "suspected": 0, "review": 0}
    for it in items:
        counts[it["status"]] += 1
    return {
        "items": items,
        "counts": counts,
        "total": len(items),
        "groups_reviewed": len(groups),
        "status_labels": MIGRATION_STATUS_LABEL,
        "ref_type_labels": REF_TYPE_LABEL,
    }
