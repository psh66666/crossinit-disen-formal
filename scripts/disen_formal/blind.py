#!/usr/bin/env python
"""Offline single-rater blind page: 160 pairs + 16 repeats.

The executor cannot visually rate images, so the page and an empty ratings.csv
are produced; the aggregate stage reports human fields as N/A until a human
fills them.  Method names and source paths are not present in the HTML.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.disen_formal.common import formal_identities, prompts, protocol, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="experiments/disen_formal_v1/protocol.json")
    ap.add_argument("--manifest", default="runs/disen_formal_v1/generated/manifest.jsonl")
    ap.add_argument("--rubric", default="experiments/disen_formal_v1/human_rubric.json")
    ap.add_argument("--out", default="results/disen_formal_v1/blind")
    args = ap.parse_args()
    proto = protocol()
    human = proto["human"]
    idents = formal_identities()
    prompt_rows = prompts()
    manifest = {}
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            manifest[(str(r["image_id"]), r["group"], str(r["prompt_id"]), int(r["seed"]))] = r["image"]

    specs = []
    for left, right, plist in (("D", "B", [p["id"] for p in prompt_rows]),
                               ("L", "B", [p["id"] for p in prompt_rows]),
                               ("D", "L", ["p03", "p04", "p05", "p06"])):
        for ident in idents:
            for pid in plist:
                specs.append((str(ident["image_id"]), left, right, pid, 101))
    if len(specs) != int(human["unique_pairs"]):
        raise SystemExit(f"built {len(specs)} pairs, expected {human['unique_pairs']}")
    rng = random.Random(int(human["blind_seed"]))
    tasks = []
    for i, (image_id, left, right, pid, seed) in enumerate(specs):
        pa = manifest[(image_id, left, pid, seed)]
        pb = manifest[(image_id, right, pid, seed)]
        if rng.random() < 0.5:
            pa, pb, left, right = pb, pa, right, left
        tasks.append({"task_id": f"t{i+1:03d}", "image_id": image_id, "prompt_id": pid,
                      "seed": seed, "A": pa, "B": pb, "groups": {"A": left, "B": right},
                      "repeat_of": None})
    ridx = list(range(len(tasks)))
    rng2 = random.Random(int(human["repeat_task_ids"].split("_")[-1]))
    chosen = rng2.sample(ridx, int(human["repeat_pairs"]))
    for j, i in enumerate(chosen):
        base = tasks[i]
        a, b = base["A"], base["B"]
        if rng2.random() < 0.5:
            a, b = b, a
        tasks.append({"task_id": f"r{j+1:03d}", "image_id": base["image_id"],
                      "prompt_id": base["prompt_id"], "seed": base["seed"],
                      "A": a, "B": b, "groups": {"A": base["groups"]["B"],
                                                 "B": base["groups"]["A"]},
                      "repeat_of": base["task_id"]})
    rng3 = random.Random(int(human["blind_seed"]) + 1)
    rng3.shuffle(tasks)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = {t["task_id"]: {"groups": t["groups"], "repeat_of": t["repeat_of"],
                          "image_id": t["image_id"], "prompt_id": t["prompt_id"]} for t in tasks}
    write_json(out / "key.json", {"protocol_id": proto["protocol_id"], "tasks": key})
    rubric = json.loads(Path(args.rubric).read_text(encoding="utf-8"))
    prompt_map = {p["id"]: p for p in prompt_rows}
    cards = []
    for t in tasks:
        p = prompt_map[t["prompt_id"]]
        rel_a = Path("../../../" + t["A"]).as_posix()
        rel_b = Path("../../../" + t["B"]).as_posix()
        source = Path(f"../../../data/disen_formal/source/{t['image_id']}.png").as_posix()
        cards.append(f"""
<article class="task" data-id="{html.escape(t['task_id'])}">
  <h3>{html.escape(t['task_id'])} · {html.escape(t['image_id'])} · {html.escape(t['prompt_id'])}</h3>
  <p class="prompt">{html.escape(p['prompt'])}</p>
  <p class="rule"><b>Success rule:</b> {html.escape(p['success_rule'])}</p>
  <div class="pair"><figure><img src="{source}"><figcaption>source</figcaption></figure>
  <figure><img src="{rel_a}"><figcaption>A</figcaption></figure>
  <figure><img src="{rel_b}"><figcaption>B</figcaption></figure></div>
  <div class="q">Identity more like source:
    <label><input type="radio" name="{t['task_id']}_id" value="A">A</label>
    <label><input type="radio" name="{t['task_id']}_id" value="B">B</label>
    <label><input type="radio" name="{t['task_id']}_id" value="tie">tie</label></div>
  <div class="q">Edit success: A
    <select name="{t['task_id']}_ea"><option value=""></option><option>yes</option><option>no</option></select>
    B <select name="{t['task_id']}_eb"><option value=""></option><option>yes</option><option>no</option></select></div>
  <div class="q">Severe artifact: A
    <select name="{t['task_id']}_aa"><option value=""></option><option>yes</option><option>no</option></select>
    B <select name="{t['task_id']}_ab"><option value=""></option><option>yes</option><option>no</option></select></div>
  <div class="q">Joint preference:
    <label><input type="radio" name="{t['task_id']}_j" value="A">A</label>
    <label><input type="radio" name="{t['task_id']}_j" value="B">B</label>
    <label><input type="radio" name="{t['task_id']}_j" value="tie">tie</label></div>
</article>""")
    page = f"""<!doctype html><meta charset="utf-8">
<title>CrossInit+Disen formal blind review</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto}}
.task{{border:1px solid #bbb;padding:1rem;margin:1.2rem 0}} .pair{{display:flex;gap:1rem}}
img{{width:360px;height:360px;object-fit:cover}} .prompt{{font-style:italic}}
.q{{margin:.5rem 0}} button{{padding:.6rem 1.2rem;font-size:1rem}}</style>
<h1>Blind review — 176 tasks</h1>
<p>Compare each pair to the source image in the course record. Method names are hidden.
Judge identity against the source, edit success by the printed rule, severe artifacts,
then joint preference. The export contains no method labels; the key is separate.</p>
<p><b>Rules:</b> {html.escape(rubric['identity'])} {html.escape(rubric['editing'])}
{html.escape(rubric['artifact'])} {html.escape(rubric['joint'])}</p>
<p><button onclick="exportCSV()">Export ratings.csv</button>
<span id="count"></span></p>
{''.join(cards)}
<script>
const tasks=[{','.join(json.dumps(t['task_id']) for t in tasks)}];
function val(n){{const e=document.querySelector(`[name="${{n}}"]:checked`);return e?e.value:''}}
function sel(n){{const e=document.querySelector(`select[name="${{n}}"]`);return e?e.value:''}}
function exportCSV(){{
 const head=['task_id','identity_preference','edit_success_A','edit_success_B','severe_artifact_A','severe_artifact_B','joint_preference'];
 const rows=[head.join(',')];
 for(const t of tasks) rows.push([t,val(t+'_id'),sel(t+'_ea'),sel(t+'_eb'),sel(t+'_aa'),sel(t+'_ab'),val(t+'_j')].join(','));
 const blob=new Blob([rows.join('\\n')],{{type:'text/csv'}}); const a=document.createElement('a');
 a.href=URL.createObjectURL(blob); a.download='ratings.csv'; a.click();
 document.getElementById('count').textContent=' exported '+tasks.length+' tasks';
}}
</script>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    with open(out / "ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "identity_preference", "edit_success_A", "edit_success_B",
                    "severe_artifact_A", "severe_artifact_B", "joint_preference"])
        for t in tasks:
            w.writerow([t["task_id"], "", "", "", "", "", ""])
    print(f"wrote {len(tasks)} tasks -> {out}/index.html, key.json, ratings.csv")


if __name__ == "__main__":
    main()
