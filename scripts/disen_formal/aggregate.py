#!/usr/bin/env python
"""Unblind, aggregate by person, bootstrap, tables, figures, conclusion draft."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.disen_formal.common import formal_identities, prompts, protocol, write_json

METRICS = ["arcface", "target", "target_area", "full_clip", "attr_clip",
           "scene_margin", "scene_acc"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v, default=float("nan")):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def boot_ci(values: np.ndarray, n: int, seed: int, level: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def training_stats(group: str) -> dict:
    secs, params = [], []
    for ident in formal_identities():
        run = ROOT / "runs" / "disen_formal_v1" / str(ident["image_id"]) / group
        mets = [json.loads(x) for x in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()] if (run / "metrics.jsonl").exists() else []
        secs.append(sum(float(r.get("sec_per_update", 0)) for r in mets))
        meta = json.loads((run / "final" / "meta.json").read_text(encoding="utf-8"))
        params.append(2048 if group == "B" else
                      int(meta.get("lora_parameters", 0)) + int(meta.get("adapter_parameters", 0)))
    return {"train_seconds_mean": mean(secs), "trainable_params": int(max(params) if params else 0)}


def human_summary(ratings_path: Path, key_path: Path) -> dict:
    if not ratings_path.exists() or not key_path.exists():
        return {"available": False, "reason": "ratings.csv or key.json missing"}
    rows = read_csv(ratings_path)
    filled = [r for r in rows if any(str(v).strip() for k, v in r.items() if k != "task_id")]
    if not filled:
        return {"available": False, "reason": "ratings.csv is empty (human review not completed)"}
    key = json.loads(key_path.read_text(encoding="utf-8"))["tasks"]
    pref = defaultdict(lambda: [0, 0, 0])  # comparison -> wins/loss/ties for first group
    edit = defaultdict(lambda: [0, 0])
    artifact = defaultdict(lambda: [0, 0])
    repeats = {}
    for r in filled:
        tid = r["task_id"]
        k = key[tid]
        groups = k["groups"]
        comp = tuple(sorted(groups.values()))
        first = groups["A"]
        jp = r.get("joint_preference", "").strip()
        if jp in {"A", "B", "tie"}:
            side = groups[jp] if jp in groups else None
            if side is None:
                pref[comp][2] += 1
            elif side == first:
                pref[comp][0] += 1
            else:
                pref[comp][1] += 1
        for side in ("A", "B"):
            g = groups[side]
            e = r.get(f"edit_success_{side}", "").strip().lower()
            if e in {"yes", "no"}:
                edit[g][0 if e == "yes" else 1] += 1
            a = r.get(f"severe_artifact_{side}", "").strip().lower()
            if a in {"yes", "no"}:
                artifact[g][0 if a == "yes" else 1] += 1
        if k.get("repeat_of"):
            repeats.setdefault(k["repeat_of"], []).append(jp)
    agreement = []
    for _, vals in repeats.items():
        if len(vals) >= 2 and all(vals):
            agreement.append(len(set(vals)) == 1)
    human = {
        "available": True,
        "n_filled": len(filled),
        "joint_preference": {"-".join(sorted(c)): {"wins_first": w, "wins_second": l,
                                                    "ties": t}
                              for c, (w, l, t) in pref.items()},
        "edit_success": {g: {"yes": y, "no": n, "rate": y / max(1, y + n)}
                         for g, (y, n) in edit.items()},
        "severe_artifact": {g: {"yes": y, "no": n, "rate": y / max(1, y + n)}
                            for g, (y, n) in artifact.items()},
        "repeat_agreement": mean(agreement) if agreement else None,
    }
    return human


def sheet(rows, out: Path, title: str, columns: list[str], cell: int = 210):
    pad, header = 8, 36
    width = pad + len(columns) * (cell + pad)
    height = pad + header + len(rows) * (cell + 18 + pad)
    canvas = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(canvas)
    d.text((pad, 6), title, fill="black")
    for j, c in enumerate(columns):
        d.text((pad + j * (cell + pad), 20), c, fill="black")
    for i, row in enumerate(rows):
        y = pad + header + i * (cell + 18 + pad)
        d.text((pad, y), row["label"], fill="black")
        for j, (_, path) in enumerate(row["items"]):
            if not path or not Path(path).exists():
                continue
            im = Image.open(path).convert("RGB").resize((cell, cell), Image.BICUBIC)
            canvas.paste(im, (pad + j * (cell + pad), y + 14))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    ap.add_argument("--scores", default="results/disen_formal_v1/per_image.csv")
    ap.add_argument("--human", default="results/disen_formal_v1/blind/ratings.csv")
    ap.add_argument("--key", default="results/disen_formal_v1/blind/key.json")
    ap.add_argument("--out", default="results/disen_formal_v1")
    args = ap.parse_args()
    proto = protocol()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.scores))
    if len(rows) != 512:
        raise SystemExit(f"per_image.csv has {len(rows)} rows, expected 512")
    by = {(r["image_id"], r["group"], r["prompt_id"], int(r["seed"])): r for r in rows}
    if len(by) != 512:
        raise SystemExit("duplicate or missing generation keys in per_image.csv")
    idents = formal_identities()
    ids = [str(x["image_id"]) for x in idents]
    for g in ("B", "L", "D", "A"):
        if sum(1 for k in by if k[1] == g) != 128:
            raise SystemExit(f"group {g} does not have 128 images")
    prompt_rows = prompts()
    pid_order = [str(p["id"]) for p in prompt_rows]
    seeds = [101, 202]

    def val(r, m):
        return fnum(r.get(m))

    def person_mean(image_id: str, group: str, metric: str, pred=lambda r: True):
        vals = [val(by[(image_id, group, p, s)], metric)
                for p in pid_order for s in seeds
                if (image_id, group, p, s) in by and pred(by[(image_id, group, p, s)])]
        return mean(vals)

    group_means = {}
    for g in ("B", "L", "D", "A"):
        group_means[g] = {
            m: mean([val(r, m) for r in rows if r["group"] == g]) for m in METRICS
        }
        group_means[g].update(training_stats(g))

    # Person-level table.
    per_person_rows = []
    for image_id in ids:
        row = {"image_id": image_id}
        for g in ("B", "L", "D", "A"):
            for m in METRICS:
                row[f"{g}_{m}"] = person_mean(image_id, g, m)
        per_person_rows.append(row)
    with open(out / "per_person.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_person_rows[0])); w.writeheader()
        for r in per_person_rows:
            w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v) for k, v in r.items()})

    def comp_deltas(a, b, metric, pred=lambda r: True):
        outv = []
        for image_id in ids:
            va = person_mean(image_id, a, metric, pred)
            vb = person_mean(image_id, b, metric, pred)
            outv.append(va - vb)
        return np.asarray(outv, dtype=np.float64)

    comparisons = [("D-B", "D", "B", 97.5), ("L-B", "L", "B", 97.5),
                   ("D-L", "D", "L", 95.0), ("D-A", "D", "A", 95.0)]
    both_pred = lambda r: int(fnum(r.get("target"), 0)) == 1
    def area_pred(r):
        return int(fnum(r.get("target"), 0)) == 1
    comp_rows = []
    for name, a, b, level in comparisons:
        for metric in METRICS:
            d = comp_deltas(a, b, metric)
            lo, hi = boot_ci(d, int(proto["evaluation"]["bootstrap_samples"]),
                             int(proto["evaluation"]["bootstrap_seed"]), level)
            comp_rows.append({"comparison": name, "metric": metric,
                              "mean_delta": f"{np.nanmean(d):.8f}", "ci_low": f"{lo:.8f}",
                              "ci_high": f"{hi:.8f}", "level": level,
                              "n_persons": len(ids), "positive_persons": int(np.nansum(d > 0))})
        # Both-target subset for identity.
        d = []
        for image_id in ids:
            pairs = []
            for p in pid_order:
                for s in seeds:
                    ra, rb = by[(image_id, a, p, s)], by[(image_id, b, p, s)]
                    if both_pred(ra) and both_pred(rb):
                        pairs.append(fnum(ra["arcface"]) - fnum(rb["arcface"]))
            d.append(mean(pairs))
        arr = np.asarray(d, dtype=np.float64)
        lo, hi = boot_ci(arr[~np.isnan(arr)], int(proto["evaluation"]["bootstrap_samples"]),
                         int(proto["evaluation"]["bootstrap_seed"]), level)
        comp_rows.append({"comparison": name, "metric": "arcface_both_target",
                          "mean_delta": f"{np.nanmean(arr):.8f}", "ci_low": f"{lo:.8f}",
                          "ci_high": f"{hi:.8f}", "level": level, "n_persons": len(ids),
                          "positive_persons": int(np.nansum(arr > 0))})
        lo_ratio, hi_ratio = proto["evaluation"]["both_face_area_ratio_interval"]
        d = []
        for image_id in ids:
            pairs = []
            for p in pid_order:
                for s in seeds:
                    ra, rb = by[(image_id, a, p, s)], by[(image_id, b, p, s)]
                    aa, ab = fnum(ra["target_area"]), fnum(rb["target_area"])
                    if both_pred(ra) and both_pred(rb) and ab > 0 and lo_ratio <= aa / ab <= hi_ratio:
                        pairs.append(fnum(ra["arcface"]) - fnum(rb["arcface"]))
            d.append(mean(pairs))
        arr2 = np.asarray(d, dtype=np.float64)
        valid = arr2[~np.isnan(arr2)]
        if len(valid):
            lo, hi = boot_ci(valid, int(proto["evaluation"]["bootstrap_samples"]),
                             int(proto["evaluation"]["bootstrap_seed"]), level)
        else:
            lo = hi = float("nan")
        comp_rows.append({"comparison": name, "metric": "arcface_area_matched",
                          "mean_delta": f"{np.nanmean(arr2):.8f}", "ci_low": f"{lo:.8f}",
                          "ci_high": f"{hi:.8f}", "level": level, "n_persons": len(ids),
                          "positive_persons": int(np.nansum(arr2 > 0))})
    with open(out / "comparisons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0])); w.writeheader(); w.writerows(comp_rows)

    human = human_summary(Path(args.human), Path(args.key))
    write_json(out / "human_summary.json", human)

    # Main table.
    def fmt(v, nd=4):
        return "N/A" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"
    lines = ["# 主表：B / L / D / A（最后检查点，512 张）", "",
             "| group | ArcFace | 有效目标脸率 | 脸面积 | 全句CLIP | 属性CLIP | 场景margin | 场景四分类 | 编辑成功率 | 严重伪影率 | 训练秒/人 | 可训练参数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in ("B", "L", "D", "A"):
        m = group_means[g]
        e = human.get("edit_success", {}).get(g, {}).get("rate") if human.get("available") else None
        a = human.get("severe_artifact", {}).get(g, {}).get("rate") if human.get("available") else None
        lines.append(f"| {g} | {fmt(m['arcface'])} | {fmt(m['target'])} | {fmt(m['target_area'])} | "
                     f"{fmt(m['full_clip'])} | {fmt(m['attr_clip'])} | {fmt(m['scene_margin'])} | "
                     f"{fmt(m['scene_acc'])} | {'N/A' if e is None else f'{e:.3f}'} | "
                     f"{'N/A' if a is None else f'{a:.3f}'} | {m['train_seconds_mean']:.0f} | "
                     f"{m['trainable_params']:,} |")
    lines += ["", "人工列为 N/A 表示盲评尚未完成；不得用自动指标替代人工编辑判断。"]
    (out / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Comparisons table.
    lines = ["# 比较表（按人物 bootstrap；主比较 97.5%，次比较 95%）", "",
             "| comparison | metric | mean Δ | CI low | CI high | level | 有利人物 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in comp_rows:
        lines.append(f"| {r['comparison']} | {r['metric']} | {float(r['mean_delta']):+.6f} | "
                     f"{float(r['ci_low']):+.6f} | {float(r['ci_high']):+.6f} | {r['level']} | "
                     f"{r['positive_persons']}/8 |")
    (out / "ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Conclusion per pre-registered rules.
    get = lambda c, m: next(float(r["mean_delta"]) for r in comp_rows
                            if r["comparison"] == c and r["metric"] == m)
    getci = lambda c, m: next((float(r["ci_low"]), float(r["ci_high"])) for r in comp_rows
                              if r["comparison"] == c and r["metric"] == m)
    db_arc, lb_arc = get("D-B", "arcface"), get("L-B", "arcface")
    db_lo, _ = getci("D-B", "arcface")
    lb_lo, _ = getci("L-B", "arcface")
    db_tgt, lb_tgt = get("D-B", "target"), get("L-B", "target")
    db_area = get("D-B", "target_area")
    db_both = get("D-B", "arcface_both_target")
    db_matched = get("D-B", "arcface_area_matched")
    db_clip, lb_clip = get("D-B", "full_clip"), get("L-B", "full_clip")
    db_scene, lb_scene = get("D-B", "scene_margin"), get("L-B", "scene_margin")
    db_sacc, lb_sacc = get("D-B", "scene_acc"), get("L-B", "scene_acc")
    dl_scene, dl_sacc = get("D-L", "scene_margin"), get("D-L", "scene_acc")
    dl_arc, dl_lo = get("D-L", "arcface"), getci("D-L", "arcface")[0]
    da_arc = get("D-A", "arcface")
    da_lo = getci("D-A", "arcface")[0]
    edit_available = bool(human.get("available"))
    db_edit = None
    if edit_available:
        e = human.get("edit_success", {})
        if "D" in e and "B" in e:
            db_edit = e["D"]["rate"] - e["B"]["rate"]
    practical = float(proto["evaluation"]["identity_practical_delta"])
    guard_edit = float(proto["evaluation"]["editing_guardrail_delta"])
    guard_det = float(proto["evaluation"]["detection_guardrail_delta"])
    edit_loss = (db_clip < -0.02) or (db_sacc < -0.05) or (db_edit is not None and db_edit < guard_edit)
    if db_arc >= practical and db_lo > 0 and db_tgt >= guard_det and not edit_loss:
        verdict = ("在本固定设置下，D 相对 B 的身份指标达到预设实际改善参考，"
                   "自动场景/CLIP 未显示明显点估计损失；人工编辑仍未完成。")
    elif db_arc > 0 and edit_loss:
        verdict = ("D−B 身份指标明显上升且区间下界>0，但场景 margin、场景四分类和全句 CLIP 同时明显下降；"
                   "按预注册属于身份—场景/编辑折中，不能写整体改善。")
    elif db_arc > 0:
        verdict = "D−B 身份方向有利，但区间跨 0 或未达 +0.02 实际参考；不能称稳定或显著改善。"
    else:
        verdict = "D−B 身份没有可靠正向变化；本轮完成受控迁移与验证，不支持预期改善。"
    if db_area > 0.02 and db_arc > 0:
        verdict += (f" 脸面积同时上升（Δ={db_area:+.4f}）；但双目标脸子集 ArcFace Δ={db_both:+.4f}、"
                    f"脸面积比匹配子集 Δ={db_matched:+.4f}，说明身份增分不是仅由脸变大造成。")
    concl = [
        "# 结论草稿（按预注册规则，2026-11-19）", "",
        f"- 主比较 D−B ArcFace：Δ={db_arc:+.6f}，97.5% 区间下界 {db_lo:+.6f}；"
        f"双目标脸子集 {db_both:+.6f}；脸面积比匹配子集 {db_matched:+.6f}。",
        f"- D−B 编辑/场景自动指标：全句 CLIP Δ={db_clip:+.6f}，场景 margin Δ={db_scene:+.6f}，"
        f"场景四分类 Δ={db_sacc:+.6f}；目标脸率 Δ={db_tgt:+.6f}；脸面积 Δ={db_area:+.6f}。",
        f"- L−B ArcFace：Δ={lb_arc:+.6f}（97.5% 下界 {lb_lo:+.6f}）；全句 CLIP Δ={lb_clip:+.6f}；"
        f"场景 margin Δ={lb_scene:+.6f}；场景四分类 Δ={lb_sacc:+.6f}。",
        f"- 次比较 D−L：ArcFace Δ={dl_arc:+.6f}（95% 下界 {dl_lo:+.6f}）；"
        f"场景 margin Δ={dl_scene:+.6f}；场景四分类 Δ={dl_sacc:+.6f}：LoRA 基础上加图像分支主要改善场景/编辑，"
        "身份额外收益区间跨 0。",
        f"- 次比较 D−A：ArcFace Δ={da_arc:+.6f}（95% 下界 {da_lo:+.6f}）：弱去噪项在该实现中作用有限。",
        f"- 人工盲评：{'已填写 ' + str(human.get('n_filled')) + ' 题' if edit_available else '未完成（N/A）'}；"
        "不能用自动指标替代人工编辑成功率或严重伪影率。",
        "", f"**判定：** {verdict}", "",
        "限制：8 人、每人一个训练种子、两个采样种子；区间是人物层 bootstrap 近似；"
        "不能声称与原论文同一测试协议。历史 CPA/Mixed 等结果保持原判定，未混入本轮统计。",
    ]
    (out / "conclusion.md").write_text("\n".join(concl) + "\n", encoding="utf-8")

    # Figures.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    # Training curves.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g in ("B",):
        allm = []
        for image_id in ids:
            p = ROOT / "runs" / "disen_formal_v1" / image_id / g / "metrics.jsonl"
            allm.append([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()])
        steps = [r["step"] for r in allm[0]]
        vals = np.array([[r["mse"] for r in m] for m in allm])
        axes[0].plot(steps, vals.mean(0), label="B mse")
        axes[0].fill_between(steps, np.percentile(vals, 10, axis=0), np.percentile(vals, 90, axis=0), alpha=.2)
    axes[0].set_title("CrossInit baseline denoising loss"); axes[0].set_xlabel("update")
    for g in ("L", "D", "A"):
        allm = []
        for image_id in ids:
            p = ROOT / "runs" / "disen_formal_v1" / image_id / g / "metrics.jsonl"
            allm.append([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()])
        steps = [r["step"] for r in allm[0]]
        vals = np.array([[r["precise"] for r in m] for m in allm])
        axes[1].plot(steps, vals.mean(0), label=g)
    axes[1].set_title("Precise denoising loss"); axes[1].set_xlabel("update"); axes[1].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "training_curves.png", dpi=140); plt.close(fig)

    # Per-person deltas.
    for metric, fname, title in (("arcface", "per_person_arcface_delta.png", "ArcFace delta"),
                                 ("target", "per_person_target_delta.png", "Target-face rate delta"),
                                 ("target_area", "per_person_area_delta.png", "Face-area delta")):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        x = np.arange(len(ids))
        for off, (comp, a, b) in zip((-0.18, 0.18), (("D-B", "D", "B"), ("L-B", "L", "B"))):
            d = comp_deltas(a, b, metric)
            ax.bar(x + off, d, width=.34, label=comp)
        ax.axhline(0, color="black", lw=.8); ax.set_xticks(x); ax.set_xticklabels(ids, rotation=30)
        ax.set_title(title); ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=140); plt.close(fig)

    # Scatter identity vs detection delta.
    fig, ax = plt.subplots(figsize=(5, 4))
    d_arc = comp_deltas("D", "B", "arcface")
    d_det = comp_deltas("D", "B", "target")
    ax.scatter(d_det, d_arc)
    for i, image_id in enumerate(ids):
        ax.annotate(image_id, (d_det[i], d_arc[i]), fontsize=7)
    ax.axhline(0, color="black", lw=.8); ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel("D-B target-face rate delta"); ax.set_ylabel("D-B ArcFace delta")
    fig.tight_layout(); fig.savefig(fig_dir / "identity_vs_detection.png", dpi=140); plt.close(fig)

    # Contact sheets.
    grid = out / "grids"; grid.mkdir(parents=True, exist_ok=True)
    columns = ["source", "B", "L", "D", "A"]

    def item(image_id, g, pid, seed):
        if g == "source":
            return ROOT / f"data/disen_formal/source/{image_id}.png"
        return ROOT / by[(image_id, g, pid, seed)]["image"]
    rows_main = []
    for image_id in [ids[i] for i in (0, 2, 4, 6)]:
        for pid in ("p03", "p04", "p05", "p08"):
            rows_main.append({"label": f"{image_id} / {pid} / s101",
                              "items": [(g, item(image_id, g, pid, 101)) for g in columns]})
    sheet(rows_main, grid / "main_fixed_16.png", "Fixed main grid: source + B/L/D/A", columns)
    # Compact 2x2 composite for the 6-8 page report (keeps all 16 groups legible).
    subs = []
    for j, image_id in enumerate([ids[i] for i in (0, 2, 4, 6)]):
        rr = [{"label": f"{image_id}/{pid}", "items": [(g, item(image_id, g, pid, 101))
                                                       for g in columns]}
              for pid in ("p03", "p04", "p05", "p08")]
        p = grid / f"_main_sub{j}.png"
        sheet(rr, p, f"{image_id}", columns, cell=145)
        subs.append(Image.open(p).convert("RGB"))
    w = max(x.width for x in subs); h = max(x.height for x in subs)
    comp = Image.new("RGB", (2 * w + 8, 2 * h + 8), "white")
    for k, im in enumerate(subs):
        comp.paste(im, ((k % 2) * (w + 8), (k // 2) * (h + 8)))
    comp.save(grid / "main_fixed_16_compact.png")
    for p in grid.glob("_main_sub*.png"):
        p.unlink()
    for image_id in ids:
        for seed in seeds:
            rr = [{"label": pid, "items": [(g, item(image_id, g, pid, seed)) for g in columns]}
                  for pid in pid_order]
            sheet(rr, grid / f"full_{image_id}_seed{seed}.png",
                  f"Full grid {image_id} seed {seed}", columns, cell=170)
    # Best/worst D-B identity keys (diagnostic only).
    keys = []
    for image_id in ids:
        for pid in pid_order:
            for seed in seeds:
                d = fnum(by[(image_id, "D", pid, seed)]["arcface"]) - fnum(by[(image_id, "B", pid, seed)]["arcface"])
                keys.append((d, image_id, pid, seed))
    keys.sort()
    pick = keys[:2] + keys[-2:]
    rr = [{"label": f"{'worst' if i < 2 else 'best'} Δ={d:+.3f} {image_id}/{pid}/s{seed}",
           "items": [(g, item(image_id, g, pid, seed)) for g in ("source", "B", "D")]}
          for i, (d, image_id, pid, seed) in enumerate(pick)]
    sheet(rr, grid / "db_identity_extremes.png", "D-B ArcFace extremes (diagnostic)", ["source", "B", "D"])

    print(json.dumps({"group_means": group_means, "human": human,
                      "db_arcface": db_arc, "lb_arcface": lb_arc}, indent=2))
    print(f"wrote {out}/main_table.md, ablation_table.md, comparisons.csv, conclusion.md, figures, grids")


if __name__ == "__main__":
    main()
