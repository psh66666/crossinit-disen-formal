# CrossInit + DisenBooth 汇报（8–10 页）

1. **任务**：CrossInit 复现 + DisenBooth 训练单元迁移；固定 8 人、8 提示、2 种子。
2. **CrossInit**：691 名字初始化两 token；81 模板；320 更新；只学两行。
3. **为什么借用 DisenBooth**：单图、身份解耦、图像 adapter + LoRA 的训练单元。
4. **四组**：B / L / D / A；D−B、L−B 主比较；D−L、D−A 解释。
5. **设置**：512、50 步、CFG 7.5、DPM-Solver++ midpoint；统一最后检查点。
6. **主表**：见 `main_table.md`；D−B ArcFace +0.1817（97.5% CI +0.1253..+0.2391）。
7. **配对图**：`grids/main_fixed_16.png`；完整 8×8×2 网格。
8. **消融**：L−B +0.1334；D−L +0.0483；D−A +0.0087。
9. **限制**：8 人、单训练种子、人工盲评未完成、自动代理选图。
10. **结论**：见 `conclusion.md`；不伪造正向结论。
