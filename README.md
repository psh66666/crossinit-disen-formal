# CrossInit + DisenBooth formal matrix (2026-11-19)

Controlled single-GPU migration of the DisenBooth training unit onto a learned
Cross Initialization condition. Eight people, eight prompts, two sampling seeds,
four groups (B/L/D/A), 512 paired images.


## Result at a glance

- **D−B ArcFace +0.1817** (97.5% person-bootstrap CI +0.1253..+0.2391), positive in
  8/8 people; both-target subset +0.1911 (100/128 pairs), face-area-matched
  subset +0.1968 (40/128 pairs). These are selected auxiliary subsets, not the
  all-128 missing-face=0 primary metric.
- **But D−B loses editing/scene ability**: full-sentence CLIP −0.0442, scene margin
  −0.0734, four-class scene accuracy −0.4219. This is an identity–scene/editing
  trade-off, **not** an overall improvement.
- L−B ArcFace +0.1334 (CI +0.0040..+0.2506), with a larger scene loss.
- The whole Disen training package (adapter + weak denoising + contrast) recovers
  scene margin +0.0405 / scene accuracy +0.3438 relative to plain LoRA, while its
  extra ArcFace gain has a 95% CI crossing zero.
- Human blind review: 176/176 completed by one rater. D−B joint preference for D
  is 0.297 [0.180,0.422] while identity preference for D is 0.859; edit success
  B/D/L = 1.000/0.583/0.042 and severe artefacts B/D/L = 0.000/0.500/0.917. The
  pre-registered −0.10 edit guardrail is breached (D−B −0.417). The repeat-key bug
  (11/16) was repaired without changing task ids, order or images.

## Known corrections

- Person IDs were recomputed with the mapping's `orig_file` column. The 10 selected
  sources remain 10 distinct people and have empty overlap with corrected historical
  persons; no pixels, weights or ArcFace scores changed, so no retraining was done.
- The adapter cosine term is a disentanglement/contrast term (positive cosine is
  minimised), not an alignment term; only comments/report wording changed.
- B's four micro-batch gradients are summed, not averaged; disclosed, no retraining.

## Papers and commits

- Cross Initialization, CVPR 2024 — commit `0b1e39a6740a0152ad78a07b5ef99c9967687ad1`.
- DisenBooth, ICLR 2024 — commit `9956ab681dbed91efdb70f9305ce43f018fa7978`.

DisenBooth's launch script value of 2000 updates is used; the paper reports about
3000. This is a course-project method transfer, not an original method and not a
bitwise reproduction.

## Environment

- Linux, one RTX 4090 24 GB, Python 3.10, PyTorch 2.1.2+cu121, diffusers 0.21.2,
  transformers 4.25.1.
- The exact installed environment is frozen in
  `experiments/disen_formal_v1/environment-installed.txt`.
- CPU ONNXRuntime is used for InsightFace scoring; BF16 is used for the U-Net.
- The SD2.1-base snapshot is the fp16-only sd2-community mirror used by the old
  project; its SHA256 is in `locks/disen_formal_v1/assets.json`.

## Data

- CelebAMask-HQ/CelebA-HQ source pool, 30000 images.
- Deterministic order: seed 20261119.
- Person IDs come from the CelebA-HQ→CelebA mapping plus `identity_CelebA.txt`.
- Historical images/persons are excluded; a post-hoc audit is in
  `data/disen_formal/history_audit_corrected.json` (empty overlap).
- Automated source screen (pose/attribute/InsightFace) is disclosed in
  `data/disen_formal/source_review.json`; human visual review is still pending.
- 8 formal + 2 reserve people, one source image each, no masks.

## Training

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
```

- B: 320 optimizer updates, microbatch 1 × accumulation 4, LR 0.005, official
  non-squared L2 anchor `1e-5`, 81 official templates, source flip p=0.5.
- L/D/A: fork from the same person's B320; frozen tokens/text encoder/base U-Net;
  2000 updates; rank-4 LoRA on all 128 attention projections, alpha 4, score 1,
  LR 1e-4; AdamW; LoRA gradient clip 1.
- D adds the 1024-d image adapter, `0.01×` weak denoising and `0.001×` cosine.
  A removes the weak term. No text-encoder LoRA, prior preservation, SNR, masks,
  or identity loss.
- All random streams are derived by SHA256(`seed:image_id:stream`) first 4 bytes
  little-endian; L/D/A share the same per-person input sequences.

## Generation and evaluation

```bash
python -m scripts.disen_formal.check coverage --stage training
python -m scripts.disen_formal.generate --all-formal --groups B L D A
python -m scripts.disen_formal.check coverage --stage generation
python -m scripts.disen_formal.blind
python -m scripts.disen_formal.score
python -m scripts.disen_formal.aggregate
python -m scripts.disen_formal.deviations
python -m scripts.disen_formal.report
python -m scripts.disen_formal.package
```

- 512 images: 8 people × 8 prompts × seeds 101/202 × B/L/D/A.
- Same explicit initial latent per key across groups; 50 steps, CFG 7.5,
  empty negative prompt, DPM-Solver++ order 2 midpoint, non-Karras linspace.
- ArcFace: buffalo_l, det 640, thresh 0.5; target face = largest qualified face,
  ties by centre then box; missing face = 0.
- CLIP eval: `openai/clip-vit-large-patch14`, L2-normalised cosine, unscaled.
- Person-level bootstrap (20000, seed 20261119); primary D−B/L−B at 97.5%,
  secondary at 95%.

## Results and evidence

- Main table: `results/disen_formal_v1/main_table.md`.
- Comparisons: `results/disen_formal_v1/comparisons.csv`.
- Per-image: `results/disen_formal_v1/per_image.csv`.
- Full grids: `results/disen_formal_v1/grids/`.
- Blind page: `results/disen_formal_v1/blind/index.html` (176 tasks).
- Report: `reports/disen_formal_v1/report.md` / `.pdf`.
- Slides: `reports/disen_formal_v1/slides.pdf`.
- Evidence archive: `crossinit_disen_formal_v1_results.tar.gz` plus manifest.

## Honest limitations

1. Human source-image review is pending. The 176-task blind review was completed
   by one rater who had seen method-labelled grids; it is a single-reviewer,
   method-name-hidden, random-left/right evaluation, not a fully blind multi-rater study.
2. Eight people, one training seed, two sampling seeds; intervals are
   person-level approximations.
3. The source selection uses an automated proxy; `source_review.png` remains for
   a human check before publication.
4. Raw faces and pretrained weights are not redistributed; follow the dataset's
   research-use terms.

## Code repository

https://github.com/psh66666/crossinit-disen-formal
