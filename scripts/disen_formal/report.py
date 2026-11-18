#!/usr/bin/env python
"""Write the 6-8 page report, PDF, slide deck PDF, and contribution note."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.disen_formal.common import protocol, write_json

OUT = ROOT / "reports" / "disen_formal_v1"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/disen_formal_v1")
    args = ap.parse_args()
    res = Path(args.results)
    main_table = (res / "main_table.md").read_text(encoding="utf-8")
    if main_table.startswith("# "):
        main_table = "\n".join(main_table.splitlines()[1:]).lstrip()
    comp_rows = read_csv(res / "comparisons.csv")
    human = json.loads((res / "human_summary.json").read_text(encoding="utf-8"))
    technical = json.loads((res / "technical_check.json").read_text(encoding="utf-8"))
    coverage = json.loads((res / "coverage.json").read_text(encoding="utf-8"))
    conclusion = (res / "conclusion.md").read_text(encoding="utf-8")
    proto = protocol()
    OUT.mkdir(parents=True, exist_ok=True)

    def row(comp, metric):
        return next(r for r in comp_rows if r["comparison"] == comp and r["metric"] == metric)
    db = row("D-B", "arcface"); lb = row("L-B", "arcface")
    dl = row("D-L", "arcface"); da = row("D-A", "arcface")
    db_t = row("D-B", "target"); db_a = row("D-B", "target_area")
    db_clip = float(row("D-B", "full_clip")["mean_delta"])
    db_scene = float(row("D-B", "scene_margin")["mean_delta"])
    db_sacc = float(row("D-B", "scene_acc")["mean_delta"])
    dl_scene = float(row("D-L", "scene_margin")["mean_delta"])
    dl_sacc = float(row("D-L", "scene_acc")["mean_delta"])
    db_pos = int(row("D-B", "arcface")["positive_persons"])
    lb_pos = int(row("L-B", "arcface")["positive_persons"])
    db_both = float(row("D-B", "arcface_both_target")["mean_delta"])
    db_match = float(row("D-B", "arcface_area_matched")["mean_delta"])

    report = f"""# CrossInit 与 DisenBooth 训练单元的单图人脸个性化：受控迁移与验证

**课程项目 · 单人 · 2026-11-19**

## 1. 任务与具体问题

本项目以 Cross Initialization（CVPR 2024）为主论文，复现其两枚可学习 token 与名字均值初始化的核心训练；再把 DisenBooth（ICLR 2024）的单图训练单元迁移到已经学到的 CrossInit 条件上。问题不是“再做一个更好看的 demo”，而是：在固定 SD2.1-base、固定 8 人、固定 8 条提示、两个采样种子的受控矩阵中，DisenBooth 式的 U-Net LoRA、图像 adapter、弱去噪与余弦对比项，是否能在**不破坏编辑能力**的前提下提高身份保真？

主目标是身份（ArcFace，缺脸记 0），同时记录有效目标脸率、脸面积、CLIP、场景诊断和人工盲评。预注册的主比较是 D−B 与 L−B；D−L 和 D−A 只解释作用，不作为额外成功门。没有开发身份、没有效果小试门，也没有根据中途损失停止正式任务。

## 2. CrossInit 核心与复现差异

CrossInit 用 691 个名字的编码结果生成两枚 token 初值：对每个名字取前两个内容 token 的 embedding，分别求均值，放入 BOS/EOS/PAD 序列后过冻结文本编码器，读取位置 1、2。训练只更新两枚 token；损失为去噪 MSE 加官方非平方 L2 正则 `1e-5 * mean(||v-v_init||_2)`。使用官方 `imagenet_templates_small` 的 81 条模板（保留顺序和重复），内部短语为 `face of <ci0> <ci1> person`；源图 bicubic 拉伸 512、水平翻转 p=0.5，无掩码、无定位损失。

复现差异必须披露：官方有效 batch 为 8，本轮为 microbatch 1×累积 4=4；VAE 后验缓存与 BF16 U-Net 是资源适配；实际权重快照是 sd2-community 的 fp16-only 镜像（与旧实验同一份权重，SHA256 已登记），不是官方 fp32 全权重；B 为 320 次更新、共 1280 个微步；所有组只使用最终检查点。文本编码器与底座冻结，两枚 token 用独立两行参数实现，优化器状态不包含整个词表；工程检查核对了梯度只进入这两行，且与官方“整表优化后恢复其他行”的两行梯度逐元素一致。

训练上，B 固定 320 次优化更新，每次更新累积 4 个单图微步，共 1280 个微步；学习率 0.005、AdamW、constant；正则项与去噪项的尺度相差约五个数量级，因此正则主要约束 token 不要远离交叉初值，而不是主导拟合。源图只做 512 bicubic 拉伸和 p=0.5 水平翻转；没有 mask、bbox、landmark 或人脸修复器，因而 B 的生成完全由两枚 token 和文本提示驱动。保存 80/160/240/320，但正式推理只读 320；中间点只用于恢复和训练曲线，不参与选择。

需要强调，B 的低 ArcFace 并不等于 CrossInit 失败：B 的优化预算只有 320 次更新，而 L/D/A 在同一个人上继续训练 2000 次；本轮比较的是“继续训练什么”，不是“CrossInit 与从零训练的公平预算对比”。这一点在解释 D−B/L−B 时必须与训练量差异一起报告。

## 3. 借用 DisenBooth 的原理、迁移与组别

DisenBooth 的训练单元包含：冻结 OpenCLIP ViT-H-14 图像特征；学习一个 sigmoid 门控 + 两层 1024 宽残差的图像 adapter；把 adapter 输出广播加到 77 个文本位置；用精确去噪、弱去噪和余弦对齐约束共同训练；U-Net 全部 attention 的 `to_q/to_k/to_v/to_out.0` 挂 rank-4 LoRA。本轮把它接到 CrossInit 的 B320 条件上，而不是替换 CrossInit 的文本条件：

- **B**：CrossInit 基线，320 更新，只学习两枚 token。
- **L**：从同一人的 B320 分叉，冻结 token 与底座，普通 U-Net LoRA，精确去噪。
- **D**：L + 图像 adapter + `0.01×`弱去噪 + `0.001×`余弦对比。
- **A**：D 去掉弱去噪项，用于 D−A 消融。

L/D/A 各 2000 更新，AdamW LR 1e-4，LoRA rank 4、alpha 4、dropout 0、scale 1，梯度裁剪 1；adapter 不额外裁剪。VAE 后验、扩散噪声、timestep 使用独立随机流，种子由 SHA256(`seed:image_id:stream`) 前 4 字节派生；同一个人 L/D/A 的输入序列逐步一致。每个任务独立进程，避免 LoRA 状态泄漏。推理不使用图像 adapter，LoRA scale 恒为 1，D/A 不用训练时的 c+u。

损失实现严格按组别取权重：L 只有精确去噪；D 为 `MSE(c+u) + 0.01·MSE(c) + 0.001·cosine(mean(c),u)`；A 去掉弱去噪项。D 为了避免保留两条大图的计算图，先对精确项和对比项反向，再重新前向计算弱项并反向；两次反向前都不 step，最后只对 LoRA 做 max_norm=1 的裁剪，adapter 不额外裁剪。工程检查用未训练的 B0 临时条件验证了：零 LoRA 时预测与未挂 LoRA 完全一致（最大差 0），D/A 的 adapter 梯度非零，检查点能恢复 optimizer/RNG，峰值 reserved 约 4.4 GiB。资源检查不产生效果证据，只证明 24 GB 单卡能按计划运行。

推理只加载父 token 与 LoRA；D/A 的图像 adapter 完全丢弃，LoRA scale 恒为 1。这样 D/A 的推理图与 L 的差别只能来自 LoRA 权重本身，而不是把训练时的 c+u 条件或图像分支偷偷带进采样。512 张图的每个 `(person,prompt,seed)` 初始 latent 都在组间逐元素相同，manifest 记录 latent、父 token、LoRA 和图片哈希。

## 4. 数据、训练、资源与评测

从 CelebAMask-HQ 的 30000 张原图按 seed=20261119 确定性排序；通过 CelebA-HQ→CelebA 映射与 `identity_CelebA.txt` 取得 person_id，排除 12 个历史图像编号及其人物、历史配置中出现过的额外图像，以及相互 ArcFace 余弦≥0.65 的疑似同人。自动筛查要求单张目标脸 det≥0.8、面积占整图 [0.06,0.50]，并利用官方 pose/attribute 注释排除严重侧转、眼镜、帽子、模糊；正式 8 人 person_id 互不相同，另冻结 2 名候补。

**需要明确披露的偏差：**执行者无法进行人工视觉审图，源图筛选采用上述自动代理并保留 `source_review.png` 供人工复核；同样，176 题盲评页面已经生成，但人工评分尚未完成，因此主表人工列为 N/A。任何自动指标都不能替代人工编辑判断。

工程检查只用第一位正式人物和 B0 未训练父条件，在独立 `engineering/` 目录跑每组 10 次更新：LoRA 零初值等效差为 {technical['zero_lora']['zero_lora_max_abs_diff']}，L/D/A timestep 序列一致={technical['rng_aligned_LDA']}，LoRA/adapter 梯度非零，检查点可恢复，峰值 reserved={technical['peak_reserved_gib']:.2f} GiB。正式训练覆盖：{sum(1 for j in coverage['jobs'] if j['ok'])}/32 个作业完成；生成 512 张，唯一键 {coverage['unique_generated_keys']}/512。

评测：ArcFace 使用 buffalo_l、det_size 640、det_thresh 0.5，合格脸 det≥0.5 且面积≥2%，取最大合格脸，并列时更居中、再按 box 字典序；缺脸身份分数记 0 并保留在 128 张/组中。CLIP 使用 `openai/clip-vit-large-patch14`，L2 归一化 cosine，不乘 2.5。场景 margin 只在 p05–p08 上计算目标场景短文本 cosine 减其余三者均值。置信区间按人物 bootstrap 20000 次，主比较报告 97.5% 区间。

## 5. 主结果与固定生成图

{main_table}

主比较：

| 比较 | 指标 | 均值差 | 区间下界 | 区间上界 |
|---|---|---:|---:|---:|
| D−B | ArcFace | {float(db['mean_delta']):+.4f} | {float(db['ci_low']):+.4f} | {float(db['ci_high']):+.4f} |
| D−B | 目标脸率 | {float(db_t['mean_delta']):+.4f} | {float(db_t['ci_low']):+.4f} | {float(db_t['ci_high']):+.4f} |
| D−B | 脸面积 | {float(db_a['mean_delta']):+.4f} | {float(db_a['ci_low']):+.4f} | {float(db_a['ci_high']):+.4f} |
| L−B | ArcFace | {float(lb['mean_delta']):+.4f} | {float(lb['ci_low']):+.4f} | {float(lb['ci_high']):+.4f} |
| D−L | ArcFace | {float(dl['mean_delta']):+.4f} | {float(dl['ci_low']):+.4f} | {float(dl['ci_high']):+.4f} |
| D−A | ArcFace | {float(da['mean_delta']):+.4f} | {float(da['ci_low']):+.4f} | {float(da['ci_high']):+.4f} |

D−B 的身份收益并不是无代价的：全句 CLIP Δ={float(db_clip):+.4f}、场景 margin Δ={float(db_scene):+.4f}、场景四分类 Δ={float(db_sacc):+.4f}，即身份明显上升时场景/编辑能力同时明显下降；按预注册只能写“身份—场景/编辑折中”，不能写整体改善。D−L 的场景 margin 为 {float(dl_scene):+.4f}、四分类为 {float(dl_sacc):+.4f}，说明图像分支/弱去噪主要把 LoRA 损失掉的场景能力补回一部分，而不是在 LoRA 之上再拿到可靠身份增益（D−L ArcFace 95% 区间跨 0）。

逐人看，D−B 的 ArcFace 差在 {db_pos}/8 人上为正，L−B 在 {lb_pos}/8 人上为正；方向一致性比单看均值更强。D−B 的双目标脸子集 Δ={db_both:+.4f}、脸面积比 [0.8,1.25] 子集 Δ={db_match:+.4f}，两者仍然为正，说明身份增分不是“脸变大导致更容易检测”单一解释。与此同时，D 的脸面积从 B 的约 0.161 升到 0.220，目标脸率从 0.844 升到 0.883（+0.039，97.5% 区间跨 0）；身份与构图同时变化，报告必须把这一起呈现，不能只报 ArcFace。

场景指标的下降在图上也很直接：B 的场景四分类接近 1.00，L 只有 0.23，D 回到 0.58，A 为 0.64。一个合理解释是：L/D/A 在单张源图上训练 2000 步，LoRA 很容易把源图的背景/构图一起记住；B 只有 320 步，尚未覆盖场景语义，因而更依赖提示。图像 adapter 与弱去噪的作用更像“把 LoRA 拉回文本条件”，它提高了 D 相对 L 的 CLIP/场景，却没有把 D 拉回 B 的水平。这是本轮最重要的失败/折中证据，而不是可以用“身份更高”掩盖的副作用。

固定正文图使用人物顺序第 1/3/5/7 人、p03/p04/p05/p08、seed101，共 16 组，不按效果挑选；完整网格在 `results/disen_formal_v1/grids/`。诊断图 `db_identity_extremes.png` 只展示 D−B 身份差最好/最差的各两个键，不替代完整网格。

![main fixed grid](results/disen_formal_v1/grids/main_fixed_16_compact.png)

![training curves](results/disen_formal_v1/figures/training_curves.png)

## 6. 消融、失败与限制

- **D−L** 回答“在 LoRA 上再加图像分支和两个目标”的净作用；**D−A** 回答弱去噪项在该实现中的作用。二者都不等于 D−B。
- B 与 L/D/A 的总训练量不同，这是两阶段方法的显式设计；D 比 L 多一条有梯度的 U-Net 前向，不能写等 FLOPs。
- 8 人、每人一个训练种子；生成有两个采样种子。人物间一致性与采样变化不能替代训练种子稳健性。
- 人工盲评未完成；主表编辑成功率/严重伪影率为 N/A。7. 结论规则要求“人眼方向一致”时，本轮不能声称联合改善。
- 本轮使用固定的常见编辑基准，部分场景与历史实验重叠；新的是人物与训练方法，不声称未见场景家族的泛化。
- 此前 CPA/Mixed 等文本侧/损失探索未达预期，保持原判定，不混入本轮统计；完整历史在附录。
- 人工盲评尚未完成，因此编辑成功率、严重伪影率和人眼偏好都没有进入主表；自动 CLIP 只能说明图文相似度，不能替代“红围巾是否真的在脖子上”“洗衣店里是否有洗衣机”这类判断。
- 本轮没有做“按效果挑检查点”：D 的 2000 步损失并不比 1500/1000 更小是可能的，但主表固定最后一步。训练曲线显示 L/D/A 的精确去噪损失高度波动，单步损失不能作为换组或早停依据。
- D−A 的弱去噪消融只解释该实现中的一个权重 0.01 项；它不证明 DisenBooth 的另一条对比目标无效，也不证明其他 rank、学习率或训练步数不会更好。本轮不追加这些搜索。
- 场景四分类在 B 上接近 1.00 而 L 上只有 0.23，这个幅度也可能部分来自 B 图像本身更接近源图背景；因此报告把场景 margin/CLIP/人工编辑一起看，不把四分类单独当成“编辑能力”的唯一定义。

![per-person arcface delta](results/disen_formal_v1/figures/per_person_arcface_delta.png)

![identity vs detection](results/disen_formal_v1/figures/identity_vs_detection.png)

**为什么仍然值得报告：**这轮不是“正号才报告”的筛选。它给出了一个可复查的受控事实：CrossInit 条件上继续训练 LoRA 能明显提高身份，但会牺牲文本编辑/场景；加入 DisenBooth 图像分支与目标能把场景损失补回一部分，代价是身份额外增益不再可靠。这比只报一个漂亮 demo 更接近课程项目要的训练单元验证。

## 7. 结论与个人贡献

{conclusion}

**个人贡献：**本人完成代码实现、工程检查、32 个训练作业、512 张配对生成、自动评测、盲评页面与报告；使用 AI 辅助完成代码编写与结果整理，按课程要求披露。单评审的盲评尚未完成，不能称独立多人研究。

## 参考

- Pang et al., Cross Initialization for Face Personalization of Text-to-Image Models, CVPR 2024, commit 0b1e39a.
- Li et al., DisenBooth: Identity-Preserving and Subject-Disentangled Diffusion Model for Personalized Text-to-Image Generation, ICLR 2024, commit 9956ab6.
- Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, CVPR 2022.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    slides = f"""# CrossInit + DisenBooth 汇报（8–10 页）

1. **任务**：CrossInit 复现 + DisenBooth 训练单元迁移；固定 8 人、8 提示、2 种子。
2. **CrossInit**：691 名字初始化两 token；81 模板；320 更新；只学两行。
3. **为什么借用 DisenBooth**：单图、身份解耦、图像 adapter + LoRA 的训练单元。
4. **四组**：B / L / D / A；D−B、L−B 主比较；D−L、D−A 解释。
5. **设置**：512、50 步、CFG 7.5、DPM-Solver++ midpoint；统一最后检查点。
6. **主表**：见 `main_table.md`；D−B ArcFace {float(db['mean_delta']):+.4f}（97.5% CI {float(db['ci_low']):+.4f}..{float(db['ci_high']):+.4f}）。
7. **配对图**：`grids/main_fixed_16.png`；完整 8×8×2 网格。
8. **消融**：L−B {float(lb['mean_delta']):+.4f}；D−L {float(dl['mean_delta']):+.4f}；D−A {float(da['mean_delta']):+.4f}。
9. **限制**：8 人、单训练种子、人工盲评未完成、自动代理选图。
10. **结论**：见 `conclusion.md`；不伪造正向结论。
"""
    (OUT / "slides.md").write_text(slides, encoding="utf-8")
    (OUT / "contribution.md").write_text(
        "# 贡献说明\n\n单人项目。本人完成方法迁移实现、工程检查、正式训练矩阵、"
        "配对生成、自动指标、盲评页面、图表与报告。AI 辅助用于代码编写、调试与结果整理；"
        "所有数值来自本轮 CSV 与日志。\n", encoding="utf-8")
    write_json(OUT / "report_assets.json", {"main_table": main_table,
                                            "conclusion": conclusion,
                                            "human_available": human.get("available", False)})
    make_pdfs()
    print(f"wrote {OUT}/report.md, slides.md, report.pdf, slides.pdf")


def _pdf_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    try:
        pdfmetrics.getFont("STSong-Light")
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def make_pdfs() -> None:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Image, Table, TableStyle)
    from reportlab.lib.utils import ImageReader
    font = _pdf_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("cn", parent=styles["BodyText"], fontName=font, fontSize=9.2,
                          leading=13.2, spaceAfter=5, wordWrap="CJK")
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=font, fontSize=15,
                        leading=20, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=font, fontSize=11.5,
                        leading=15, spaceAfter=5)
    from xml.sax.saxutils import escape
    def txt(s: str) -> str:
        return escape(s.replace("**", ""))
    report = (OUT / "report.md").read_text(encoding="utf-8")
    story = []
    lines = report.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            story.append(Spacer(1, 2)); i += 1; continue
        if s.startswith("# "):
            story.append(Paragraph(s[2:], h1)); i += 1; continue
        if s.startswith("## "):
            story.append(Paragraph(s[3:], h2)); i += 1; continue
        if s.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                t = lines[i].strip()
                if not (set(t.replace("|", "").strip()) <= {"-", ":", " "}):
                    table_lines.append([c.strip() for c in t.strip("|").split("|")])
                i += 1
            if table_lines:
                ncol = max(len(r) for r in table_lines)
                data = [[Paragraph(txt(c), body) for c in r] for r in table_lines]
                tbl = Table(data, colWidths=[17.0 * cm / ncol] * ncol)
                tbl.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(tbl); story.append(Spacer(1, 5))
            continue
        if s.startswith("!["):
            try:
                rel = s.split("](", 1)[1].rstrip(")")
                p = ROOT / rel
                if p.exists():
                    iw, ih = ImageReader(str(p)).getSize()
                    width = 16.5 * cm if "grid" not in rel else 13.0 * cm
                    height = width * ih / iw
                    if height > 19 * cm:
                        height = 19 * cm; width = height * iw / ih
                    story.append(Image(str(p), width=width, height=height))
                    story.append(Spacer(1, 4))
            except Exception:
                pass
            i += 1; continue
        story.append(Paragraph(txt(s), body)); i += 1
    doc = SimpleDocTemplate(str(OUT / "report.pdf"), pagesize=A4,
                            leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="CrossInit + DisenBooth report")
    doc.build(story)

    slide_lines = (OUT / "slides.md").read_text(encoding="utf-8").splitlines()
    slides = [x for x in slide_lines if x.strip() and not x.startswith("#")]
    deck = SimpleDocTemplate(str(OUT / "slides.pdf"), pagesize=landscape(A4),
                             leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                             topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                             title="CrossInit + DisenBooth slides")
    import re
    story = [Paragraph("CrossInit + DisenBooth：受控迁移与验证", h1)]
    for line in slides:
        story.append(Paragraph(txt(line), body))
        m = re.match(r"^(\d+)\.", line.strip())
        if m:
            story.append(Spacer(1, 6))
            if m.group(1) != "10":
                story.append(PageBreak())
    deck.build(story)


if __name__ == "__main__":
    main()
