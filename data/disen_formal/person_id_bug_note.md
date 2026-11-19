# person_id 列勘误

`candidate_inventory.csv` 的 `person_id` 与早期 `identities.json` 的 `person_id`
由 `CelebA-HQ-to-CelebA-mapping.txt` 的 `orig_idx`（0 基）直接索引
`identity_CelebA.txt` 得到；官方文件名是 `orig_file`（1 基），因此这些编号整体错位。

不影响：

- 源图像素、512 处理、VAE/CLIP 缓存；
- 训练权重、生成图、ArcFace/CLIP 分数；
- 每张源图的 ArcFace 目标身份本身。

已修正：

- `identities.json` 增加 `person_id`（orig_file 正确值）、`person_id_at_training`
  （错位值）、`celeb_orig_idx`、`celeb_orig_file`；
- `person_id_mapping_audit.csv` 列出 10 个源图的完整映射；
- `history_audit_corrected.json` 用正确编号重算历史人物集合，overlap 为空；
- 10 个正式+候补源图仍是 10 个互不相同的人物。

`candidate_inventory.csv` 是当时的筛选日志，未回填正确编号；任何人物层统计/审计应以
`identities.json` 和 `person_id_mapping_audit.csv` 为准。
