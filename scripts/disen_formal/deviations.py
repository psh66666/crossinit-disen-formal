#!/usr/bin/env python
"""Write the human-readable protocol-deviation log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="results/disen_formal_v1/human_summary.json")
    ap.add_argument("--out", default="results/disen_formal_v1/deviations.jsonl")
    args = ap.parse_args()
    human = {}
    p = ROOT / args.human
    if p.exists():
        human = json.loads(p.read_text(encoding="utf-8"))
    rows = [
        {"id": "automated_source_review",
         "what": "Source images were screened by pose/attribute/InsightFace proxy, not human visual review.",
         "why": "The executor cannot see images. source_review.png remains available for a human check.",
         "impact": "Formal identities could still contain an image a human would reject."},
        {"id": "human_blind_review_not_completed",
         "what": "176-task blind page and empty ratings.csv were produced; no labels were entered.",
         "why": "Human visual judgment cannot be automated honestly.",
         "impact": "Editing success, severe-artifact rate and human preference are N/A in the main table.",
         "ratings_available": bool(human.get("available", False))},
        {"id": "fp16_model_snapshot",
         "what": "The SD2.1-base snapshot is the fp16-only sd2-community mirror already used by the old project.",
         "why": "It is the frozen weight set bound by the protocol; SHA256 hashes are in assets/pretrained hashes.",
         "impact": "Not a byte-identical copy of the official fp32 release."},
        {"id": "effective_batch_4",
         "what": "B uses microbatch 1 x accumulation 4 instead of the official effective batch 8.",
         "why": "4090 memory/throughput adaptation, fixed before training.",
         "impact": "Not claimed as bitwise reproduction of the paper."},
        {"id": "adaptation_updates_2000",
         "what": "L/D/A use 2000 updates, the official launch-script value; the paper reports about 3000.",
         "why": "Pre-registered fixed budget; no 1000/2000/3000 search.",
         "impact": "Disclosed in the report; no post-hoc checkpoint selection."},
        {"id": "engineering_b0_parent",
         "what": "The engineering check used the untrained cross-initialized condition (parent_step 0), not B320.",
         "why": "Check only shapes/gradients/memory before paying for 8 baselines.",
         "impact": "No method-effect evidence is taken from engineering outputs."},
        {"id": "history_audit_post_hoc",
         "what": "Extra old-project image ids were audited after selection; overlap with the selected persons was empty.",
         "why": "The protocol's 12 ids were applied before selection; the old processed directory contained additional ids.",
         "impact": "No reselection needed; data/disen_formal/history_audit.json records the check."},
        {"id": "duplicate_launcher_race",
         "what": "Two launchers briefly started the same L/D/A jobs because an automatic chained launcher overlapped with a manual one.",
         "why": "Operational mistake in process supervision, not a method change.",
         "impact": "All L/D/A outputs from the affected few minutes were deleted before any result inspection; "
                   "the 24 jobs were restarted once from clean directories. B outputs were unaffected."},
        {"id": "post_freeze_analysis_changes",
         "what": "After the freeze, only analysis/inference-support code changed: bf16-safe tensor hashing, "
                 "a generate --image-ids filter for smoke tests/targeted regeneration, and report/aggregate scripts.",
         "why": "These are logging/tooling fixes; no training loss, optimizer, data, or model computation changed.",
         "impact": "The formal training runs were not restarted; the freeze's training-math assets still apply."},
        {"id": "prompt_reuse",
         "what": "The eight prompts are fixed common edit prompts; some scenes overlap earlier experiments.",
         "why": "The new element is the people and the training method, not unseen prompt families.",
         "impact": "No claim of generalization to unseen scene families."},
    ]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    print(f"wrote {len(rows)} deviations -> {out}")


if __name__ == "__main__":
    main()
