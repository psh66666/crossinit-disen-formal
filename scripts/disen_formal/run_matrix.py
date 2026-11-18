#!/usr/bin/env python
"""Serial single-GPU matrix with safe re-entry.

No metric-based stopping.  A failed job is recorded and the matrix continues
with the next independent person; the exit code is non-zero so it is visible.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossinit.disen_state import is_complete
from scripts.disen_formal.common import formal_identities, protocol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    ap.add_argument("--freeze", default="locks/disen_formal_v1/freeze.json")
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--all-formal", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--engineering", action="store_true")
    args = ap.parse_args()
    idents = formal_identities()
    order = [str(x["image_id"]) for x in idents]
    groups = [g.upper() for g in args.groups]
    if set(groups) - {"B", "L", "D", "A"}:
        raise SystemExit(f"bad groups {groups}")
    logdir = ROOT / ("engineering/disen_formal_v1" if args.engineering else "runs/disen_formal_v1")
    logdir.mkdir(parents=True, exist_ok=True)
    matrix_log = logdir / "matrix.log"
    failures = []
    for pos, image_id in enumerate(order):
        if groups == ["B"]:
            job_groups = ["B"]
        else:
            job_groups = groups if pos % 2 == 0 else list(reversed(groups))
        for group in job_groups:
            run_dir = logdir / image_id / group
            if not args.engineering and is_complete(run_dir) and args.resume:
                print(f"[skip complete] {image_id}/{group}")
                continue
            cmd = [sys.executable, "-m", "scripts.disen_formal.train",
                   "--protocol", args.protocol, "--freeze", args.freeze,
                   "--image-id", image_id, "--group", group]
            if args.resume:
                cmd.append("--resume")
            if args.engineering:
                cmd.append("--engineering")
            line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} START {image_id}/{group} {' '.join(cmd)}"
            print(line, flush=True)
            with open(matrix_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            started = time.time()
            proc = subprocess.run(cmd, cwd=ROOT)
            elapsed = time.time() - started
            status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
            line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} END {image_id}/{group} {status} {elapsed:.1f}s"
            print(line, flush=True)
            with open(matrix_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if proc.returncode != 0:
                failures.append({"image_id": image_id, "group": group,
                                 "returncode": proc.returncode, "elapsed_sec": round(elapsed, 1)})
    out = logdir / "matrix_failures.json"
    out.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(f"{len(failures)} job(s) failed; see {out}")
        raise SystemExit(1)
    print("matrix complete; 0 failures")


if __name__ == "__main__":
    main()
