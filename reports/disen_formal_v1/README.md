# CrossInit + DisenBooth formal v1 — evidence package

This directory is the deliverable for the 2026-11-19 formal matrix.

## Result at a glance

- D−B ArcFace **+0.1817** (97.5% person-bootstrap CI +0.1253..+0.2391), positive in 8/8
  people; the both-target subset is +0.1911 (100/128 pairs) and the face-area-matched
  subset +0.1968 (40/128 pairs). These are selected auxiliary subsets, not randomised,
  and do not replace the all-128 missing-face=0 primary metric.
- But D−B also loses editing/scene ability: full-sentence CLIP −0.0442, scene margin
  −0.0734, four-class scene accuracy −0.4219. This is an **identity–scene/editing
  trade-off**, not an overall improvement.
- L−B ArcFace +0.1334 (CI +0.0040..+0.2506), with an even larger scene loss.
- D−L improves scene margin +0.0405 and scene accuracy +0.3438, while the extra
  ArcFace gain has a 95% CI crossing zero. The whole Disen training package
  (adapter + weak denoising + contrast) mainly recovers scene ability lost by plain LoRA.
- Human blind ratings are still in progress; editing success/artefact columns remain
  **N/A** until all 176 tasks are complete.

## Known corrections before final scoring

- The blind repeat-key bug (11/16 repeats) was repaired from the HTML image paths;
  task ids, page order and images were unchanged. A corrected human aggregator now
  uses 160 originals for effects and 16 repeats only for consistency.
- Person IDs were recomputed with the mapping's `orig_file`; the 10 selected sources
  remain 10 distinct people and have empty overlap with corrected historical persons.
  No pixels, weights or ArcFace scores changed, so no retraining was done.
- The adapter cosine term is a disentanglement/contrast term (positive cosine is
  minimised); comments and report wording were corrected, weights were not changed.
- B's four micro-batch gradients are summed, not averaged; this is disclosed.

## What is inside

- `report.md` / `report.pdf`: 6–8 page report.
- `slides.pdf` / `slides.md`: 8–10 page talk deck.
- `contribution.md`: single-person contribution and AI-assistance disclosure.
- `results/disen_formal_v1/`:
  - `per_image.csv` — 512 rows, one per (person, group, prompt, seed).
  - `per_person.csv` — person-level means.
  - `main_table.md`, `ablation_table.md`, `comparisons.csv` — main/ablation tables
    and person-level bootstrap intervals.
  - `conclusion.md` — pre-registered interpretation, including N/A human fields.
  - `figures/` — training curves, per-person deltas, scatter.
  - `grids/` — fixed 16-group sheet and full 8×8×2 grids.
  - `blind/index.html` + `key.json` + `ratings.csv` — 176 blind tasks; the empty
    ratings file means human review was **not** completed by the executor.
- `runs/disen_formal_v1/<image_id>/<B|L|D|A>/` — config, logs, metrics, final
  checkpoints; optimizer/RNG checkpoints are included in the full archive.
- `runs/disen_formal_v1/generated/manifest.jsonl` — all 512 image keys, hashes,
  initial-latent hashes, parent token/LoRA hashes.
- `data/disen_formal/` — candidate inventory, source review, identities, 512
  source PNGs, VAE/CLIP caches and random plans.
- `locks/disen_formal_v1/` — environment, asset hashes, freeze lock.
- `engineering/` is intentionally **not** included: it is a discarded resource
  check, not method evidence.

## Reproduction entry points

```bash
python -m scripts.disen_formal.assets register --protocol experiments/disen_formal_v1/protocol.json \
  --discover-under /root/autodl-tmp --out locks/disen_formal_v1/assets.json
python -m scripts.disen_formal.data candidates --protocol experiments/disen_formal_v1/protocol.json \
  --assets locks/disen_formal_v1/assets.json --out data/disen_formal
python -m scripts.disen_formal.data finalize --protocol experiments/disen_formal_v1/protocol.json \
  --review data/disen_formal/source_review.json --out data/disen_formal/identities.json
python -m scripts.disen_formal.cache prepare
python -m scripts.disen_formal.check engineering --protocol experiments/disen_formal_v1/protocol.json \
  --out results/disen_formal_v1/technical_check.json
python -m scripts.disen_formal.assets freeze --protocol experiments/disen_formal_v1/protocol.json \
  --identities data/disen_formal/identities.json \
  --engineering results/disen_formal_v1/technical_check.json \
  --out locks/disen_formal_v1/freeze.json
python -m scripts.disen_formal.run_matrix --protocol experiments/disen_formal_v1/protocol.json \
  --freeze locks/disen_formal_v1/freeze.json --groups B --all-formal --resume
python -m scripts.disen_formal.cache text --all-formal
python -m scripts.disen_formal.run_matrix --protocol experiments/disen_formal_v1/protocol.json \
  --freeze locks/disen_formal_v1/freeze.json --groups L D A --all-formal --resume
python -m scripts.disen_formal.check coverage --stage training
python -m scripts.disen_formal.generate --all-formal --groups B L D A
python -m scripts.disen_formal.blind
python -m scripts.disen_formal.score
python -m scripts.disen_formal.aggregate
python -m scripts.disen_formal.report
python -m scripts.disen_formal.package
```

## Honest limitations

1. Source images were selected with an automated pose/attribute/InsightFace proxy.
   `source_review.png` still requires a human visual check before publication.
2. The 176-task blind page is being completed by one human rater; final labels
   were not available when this package snapshot was made. The report's human
   editing/artifact columns are **N/A** until completion; automatic ArcFace/CLIP
   scores are not a substitute.
3. Eight people, one training seed per person, two sampling seeds. Bootstrap
   intervals over people are approximate and cannot replace training-seed
   robustness.
4. DisenBooth's official launch script uses 2000 updates; the paper reports about
   3000. This difference is disclosed and not searched.
5. The raw face dataset and pretrained weights are not redistributed. Keep them
   under the dataset's research-use terms.
