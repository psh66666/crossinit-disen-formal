#!/usr/bin/env python
"""Build the evidence archive and verify counts/hashes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.disen_state import sha256_file, write_json
from scripts.disen_formal.common import protocol


def add(tar: tarfile.TarFile, path: Path, arc: str) -> int:
    n = 0
    if not path.exists():
        return 0
    for p in sorted(path.rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp"):
            tar.add(p, arcname=str(Path(arc) / p.relative_to(path)))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    ap.add_argument("--freeze", default="locks/disen_formal_v1/freeze.json")
    ap.add_argument("--out", default="crossinit_disen_formal_v1_results.tar.gz")
    ap.add_argument("--results-only", action="store_true",
                    help="exclude optimizer checkpoints and generated images (small archive)")
    args = ap.parse_args()
    proto = protocol()
    ident = json.loads((ROOT / "data" / "disen_formal" / "identities.json").read_text(encoding="utf-8"))
    formal = [x for x in ident["identities"] if x["status"] == "formal"]
    if len(formal) != 8:
        raise SystemExit("need 8 formal identities before packaging")
    manifest = ROOT / "runs" / "disen_formal_v1" / "generated" / "manifest.jsonl"
    ok = 0
    if manifest.exists():
        ok = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines()
                 if line.strip() and json.loads(line).get("status") == "ok")
    if ok != 512:
        raise SystemExit(f"manifest has {ok} ok images, expected 512")
    blind_key = ROOT / "results" / "disen_formal_v1" / "blind" / "key.json"
    if not blind_key.exists():
        raise SystemExit("blind/key.json missing")
    k = json.loads(blind_key.read_text(encoding="utf-8"))["tasks"]
    if len(k) != 176:
        raise SystemExit(f"blind tasks={len(k)}, expected 176")

    out = Path(args.out)
    counts = {}
    arcroot = "disen_formal_v1_20261119"
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        counts["code"] = 0
        for rel in ("crossinit", "scripts/disen_formal"):
            counts[f"code/{rel}"] = add(tar, ROOT / rel, f"{arcroot}/{rel}")
        counts["experiments"] = add(tar, ROOT / "experiments" / "disen_formal_v1",
                                    f"{arcroot}/experiments/disen_formal_v1")
        counts["third_party"] = add(tar, ROOT / "third_party" / "official",
                                    f"{arcroot}/third_party/official")
        counts["locks"] = add(tar, ROOT / "locks" / "disen_formal_v1",
                              f"{arcroot}/locks/disen_formal_v1")
        counts["data_meta"] = 0
        for rel in ("identities.json", "candidate_inventory.csv", "candidate_pool.json",
                    "candidate_order.json", "source_review.json", "source_review.png",
                    "history_audit.json", "history_audit_corrected.json", "person_id_mapping_audit.csv", "person_id_bug_note.md",
                    "celeb_init_691.pt", "random_plans"):
            p = ROOT / "data" / "disen_formal" / rel
            if p.is_file():
                tar.add(p, arcname=f"{arcroot}/data/disen_formal/{rel}"); counts["data_meta"] += 1
            elif p.is_dir():
                counts["data_meta"] += add(tar, p, f"{arcroot}/data/disen_formal/{rel}")
        counts["source"] = add(tar, ROOT / "data" / "disen_formal" / "source",
                               f"{arcroot}/data/disen_formal/source")
        counts["cache"] = add(tar, ROOT / "data" / "disen_formal" / "cache",
                              f"{arcroot}/data/disen_formal/cache")
        runs = ROOT / "runs" / "disen_formal_v1"
        for p in sorted(runs.glob("*/*/final")):
            rel = p.relative_to(runs)
            counts[f"final/{rel}"] = add(tar, p, f"{arcroot}/runs/disen_formal_v1/{rel}")
        for p in sorted(runs.glob("*/*/config.json")):
            rel = p.relative_to(runs)
            tar.add(p, arcname=f"{arcroot}/runs/disen_formal_v1/{rel}")
        for p in sorted(runs.glob("*/*/train.log")):
            rel = p.relative_to(runs)
            tar.add(p, arcname=f"{arcroot}/runs/disen_formal_v1/{rel}")
        for p in sorted(runs.glob("*/*/metrics.jsonl")):
            rel = p.relative_to(runs)
            tar.add(p, arcname=f"{arcroot}/runs/disen_formal_v1/{rel}")
        for p in sorted(runs.glob("*/*/status.json")):
            rel = p.relative_to(runs)
            tar.add(p, arcname=f"{arcroot}/runs/disen_formal_v1/{rel}")
        if not args.results_only:
            counts["checkpoints"] = 0
            for p in sorted(runs.glob("*/*/checkpoints")):
                counts["checkpoints"] += add(tar, p,
                                             f"{arcroot}/runs/disen_formal_v1/{p.relative_to(runs)}")
            counts["generated"] = add(tar, runs / "generated",
                                       f"{arcroot}/runs/disen_formal_v1/generated")
        counts["results"] = add(tar, ROOT / "results" / "disen_formal_v1",
                                f"{arcroot}/results/disen_formal_v1")
        counts["reports"] = add(tar, ROOT / "reports" / "disen_formal_v1",
                                f"{arcroot}/reports/disen_formal_v1")
        readme = ROOT / "reports" / "disen_formal_v1" / "README.md"
        if readme.exists():
            tar.add(readme, arcname=f"{arcroot}/README.md")
    meta = {"protocol_id": proto["protocol_id"], "archive": str(out),
            "bytes": out.stat().st_size, "sha256": sha256_file(out), "files": counts,
            "generated_ok": ok, "blind_tasks": len(k)}
    write_json(Path(str(out) + ".manifest.json"), meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
