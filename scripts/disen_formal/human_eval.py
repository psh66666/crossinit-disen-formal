"""Human blind-rating aggregation with the corrected task/key semantics.

Rules fixed after the 2026-11-19 audit and before completing the ratings:

- 160 original tasks contribute to preference effects; 16 repeats only to
  consistency, never as extra samples.
- Preferences are translated from A/B sides to method labels using the key.
- Edit/artefact labels are deduplicated by (image_id, group, prompt_id, seed),
  keeping the first label in the page's fixed presentation order; conflicts are
  reported, never resolved by choosing a favourable label.
- The primary editing/artefact rates use p03-p08, seed 101, 48 unique images per
  group. p01/p02 are auxiliary.
- Per-person means, then a person bootstrap (20,000, seed 20261119, 95% central
  interval). This is a descriptive interval; the original protocol did not
  specify a human CI, so it is locked here before scoring is complete.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PREF_FIELDS = ("identity_preference", "joint_preference")
SIDE_FIELDS = ("edit_success_A", "edit_success_B", "severe_artifact_A", "severe_artifact_B")
TASK_FIELDS = PREF_FIELDS + SIDE_FIELDS
PRIMARY_PROMPTS = {"p03", "p04", "p05", "p06", "p07", "p08"}
NEUTRAL_PROMPTS = {"p01", "p02"}
PRIMARY_SEED = 101


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pref(v: str) -> str | None:
    v = (v or "").strip()
    if v in {"A", "B", "tie"}:
        return v
    return None


def _yesno(v: str) -> str | None:
    v = (v or "").strip().lower()
    return v if v in {"yes", "no"} else None


def _mean(xs):
    xs = [float(x) for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _boot(values: np.ndarray, samples: int, seed: int, level: float = 95.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, (100 - level) / 2)), float(np.percentile(means, (100 + level) / 2))


def task_method_pref(row: dict, entry: dict, field: str) -> str:
    """Return the preferred method group, 'tie', or 'invalid'."""
    p = _pref(row.get(field, ""))
    if p is None:
        return "invalid"
    if p == "tie":
        return "tie"
    return entry["groups"][p]


def _comparison_name(groups: dict) -> str:
    return "-".join(sorted(groups.values()))


def _target_for_comp(comp: str) -> str | None:
    return {"B-D": "D", "B-L": "L", "D-L": "D"}.get(comp)


def _rate_side(labels: dict, group: str) -> dict:
    yes, no = labels["yes"], labels["no"]
    return {"yes": yes, "no": no, "n": yes + no,
            "rate": yes / (yes + no) if yes + no else float("nan")}


def _collect_image_labels(rows_by_task: dict, order: list[str], key: dict):
    edit = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    artifact = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    first_source = {}
    conflicts = []
    for tid in order:
        row = rows_by_task.get(tid)
        if row is None:
            continue
        entry = key[tid]
        for side in ("A", "B"):
            g = entry["groups"][side]
            image_key = entry.get(f"{side}_key") or f"{entry['image_id']}|{g}|{entry['prompt_id']}|{entry['seed']}"
            pid = entry["prompt_id"]
            e = _yesno(row.get(f"edit_success_{side}", ""))
            a = _yesno(row.get(f"severe_artifact_{side}", ""))
            if image_key in first_source:
                # A repeated image must not contribute a second sample.
                if e is not None:
                    prev = first_source[image_key].get("edit")
                    if prev is not None and prev != e:
                        conflicts.append({"image_key": image_key, "field": "edit",
                                          "kept": prev, "later": e, "later_task": tid})
                if a is not None:
                    prev = first_source[image_key].get("artifact")
                    if prev is not None and prev != a:
                        conflicts.append({"image_key": image_key, "field": "artifact",
                                          "kept": prev, "later": a, "later_task": tid})
                continue
            first_source[image_key] = {"edit": e, "artifact": a, "group": g,
                                       "prompt_id": pid, "task": tid}
            if e is not None:
                edit[g][pid][0 if e == "yes" else 1] += 1
            if a is not None:
                artifact[g][pid][0 if a == "yes" else 1] += 1
    return edit, artifact, first_source, conflicts


def _rates_by_group(by_prompt: dict, prompts: set[str]) -> dict:
    out = {}
    for g, per_prompt in by_prompt.items():
        yes = sum(per_prompt[p][0] for p in prompts if p in per_prompt)
        no = sum(per_prompt[p][1] for p in prompts if p in per_prompt)
        out[g] = {"yes": yes, "no": no, "n": yes + no,
                  "rate": yes / (yes + no) if yes + no else float("nan")}
    return out


def _pref_block(rows_by_task: dict, order: list[str], key: dict, comp: str,
                field: str, target: str | None, samples: int, seed: int) -> dict:
    """Per-person win values for `target` in a fixed comparison; repeats excluded."""
    per_person = defaultdict(list)
    wins = losses = ties = 0
    for tid in order:
        entry = key[tid]
        if entry.get("repeat_of"):
            continue
        if _comparison_name(entry["groups"]) != comp:
            continue
        row = rows_by_task.get(tid)
        if row is None:
            continue
        p = _pref(row.get(field, ""))
        if p is None:
            continue
        if p == "tie":
            value = 0.5
            ties += 1
        elif entry["groups"][p] == target:
            value = 1.0
            wins += 1
        else:
            value = 0.0
            losses += 1
        per_person[entry["image_id"]].append(value)
    person_means = np.asarray([_mean(v) for v in per_person.values()], dtype=float)
    lo, hi = _boot(person_means, samples, seed)
    return {"target": target, "n_pairs": wins + losses + ties,
            "wins": wins, "losses": losses, "ties": ties,
            "win_rate": _mean([1.0] * wins + [0.0] * losses + [0.5] * ties),
            "person_mean": float(np.nanmean(person_means)) if len(person_means) else float("nan"),
            "ci_low": lo, "ci_high": hi,
            "per_person": {k: _mean(v) for k, v in per_person.items()}}


def _dedupe_agreement(rows_by_task: dict, key: dict,
                      repeats_meta: dict[str, dict]) -> dict:
    """Compare repeat preference with its original after translating to methods."""
    details = []
    agree = {"identity": 0, "joint": 0}
    valid = {"identity": 0, "joint": 0}
    for rid, meta in repeats_meta.items():
        orig = meta["repeat_of"]
        if rid not in rows_by_task or orig not in rows_by_task:
            continue
        rrow, orow = rows_by_task[rid], rows_by_task[orig]
        detail = {"repeat": rid, "original": orig}
        for field, short in (("identity_preference", "identity"), ("joint_preference", "joint")):
            a = task_method_pref(orow, key[orig], field)
            b = task_method_pref(rrow, key[rid], field)
            detail[short] = {"original": a, "repeat": b}
            if a != "invalid" and b != "invalid":
                valid[short] += 1
                # A repeat is consistent if both choose the same method, or both tie.
                if a == b:
                    agree[short] += 1
        details.append(detail)
    return {"n": len(details), "valid_identity": valid["identity"],
            "valid_joint": valid["joint"],
            "identity_agreement": agree["identity"] / valid["identity"] if valid["identity"] else None,
            "joint_agreement": agree["joint"] / valid["joint"] if valid["joint"] else None,
            "details": details}


def human_summary(ratings_path: Path, key_path: Path, bootstrap_samples: int = 20000,
                  bootstrap_seed: int = 20261119) -> dict:
    if not ratings_path.exists() or not key_path.exists():
        return {"available": False, "status": "empty",
                "reason": "ratings.csv or key.json missing"}
    key_obj = json.loads(Path(key_path).read_text(encoding="utf-8"))
    key = key_obj["tasks"]
    order = key_obj.get("order") or sorted(key)
    rows = read_csv(Path(ratings_path))
    by_task = {r["task_id"]: r for r in rows if r.get("task_id") in key}
    invalid = {}
    complete = 0
    for tid in order:
        row = by_task.get(tid)
        if row is None:
            invalid[tid] = ["missing_row"]
            continue
        bad = []
        for f in PREF_FIELDS:
            if _pref(row.get(f, "")) is None:
                bad.append(f)
        for f in SIDE_FIELDS:
            if _yesno(row.get(f, "")) is None:
                bad.append(f)
        if bad:
            invalid[tid] = bad
        else:
            complete += 1
    status = "complete" if complete == len(order) else ("partial" if complete else "empty")
    if status == "empty":
        return {"available": False, "status": "empty",
                "reason": "ratings.csv has no complete task",
                "n_tasks": len(order), "n_complete": 0}

    # Preferences: only the 160 original tasks; repeats are consistency only.
    repeats = {tid: e for tid, e in key.items() if e.get("repeat_of")}
    preference = {}
    for comp in ("B-D", "B-L", "D-L"):
        target = _target_for_comp(comp)
        if target is None:
            continue
        block = {}
        for field in PREF_FIELDS:
            block[field.replace("_preference", "")] = _pref_block(
                by_task, order, key, comp, field, target, bootstrap_samples, bootstrap_seed)
        preference[comp] = block

    edit_by_prompt, artifact_by_prompt, first_source, conflicts = _collect_image_labels(
        by_task, order, key)
    edit_primary = _rates_by_group(edit_by_prompt, PRIMARY_PROMPTS)
    edit_neutral = _rates_by_group(edit_by_prompt, NEUTRAL_PROMPTS)
    artifact_primary = _rates_by_group(artifact_by_prompt, PRIMARY_PROMPTS)
    artifact_all = _rates_by_group(artifact_by_prompt,
                                   PRIMARY_PROMPTS | NEUTRAL_PROMPTS)

    # Per-person edit/artefact rates for D-B and L-B primary guardrail.
    per_person_edit = defaultdict(lambda: defaultdict(list))
    for image_key, rec in first_source.items():
        if rec["prompt_id"] not in PRIMARY_PROMPTS or rec["edit"] is None:
            continue
        per_person_edit[rec["group"]][_task_image_id(image_key)].append(1.0 if rec["edit"] == "yes" else 0.0)
    edit_diffs = {}
    for comp, a, b in (("D-B", "D", "B"), ("L-B", "L", "B")):
        ids = sorted(set(per_person_edit[a]) | set(per_person_edit[b]))
        vals = []
        for i in ids:
            va = _mean(per_person_edit[a].get(i, []))
            vb = _mean(per_person_edit[b].get(i, []))
            if np.isfinite(va) and np.isfinite(vb):
                vals.append(va - vb)
        vals = np.asarray(vals)
        lo, hi = _boot(vals, bootstrap_samples, bootstrap_seed)
        edit_diffs[comp] = {"mean_delta": float(np.nanmean(vals)) if len(vals) else float("nan"),
                            "ci_low": lo, "ci_high": hi, "n_persons": int(np.isfinite(vals).sum())}

    summary = {
        "available": status == "complete",
        "status": status,
        "n_tasks": len(order),
        "n_complete": complete,
        "n_filled": complete,
        "invalid": invalid,
        "rater": "single rater; saw method-labelled grids before this blind pass",
        "preference": preference,
        "edit_success": {
            g: {**edit_primary.get(g, {"yes": 0, "no": 0, "n": 0, "rate": float("nan")}),
                "rate": edit_primary.get(g, {}).get("rate", float("nan")),
                "neutral_rate": edit_neutral.get(g, {}).get("rate", float("nan")),
                "neutral_n": edit_neutral.get(g, {}).get("n", 0)}
            for g in ("B", "L", "D", "A")
        },
        "severe_artifact": {
            g: {**artifact_primary.get(g, {"yes": 0, "no": 0, "n": 0, "rate": float("nan")}),
                "rate": artifact_primary.get(g, {}).get("rate", float("nan")),
                "all_prompt_rate": artifact_all.get(g, {}).get("rate", float("nan")),
                "all_prompt_n": artifact_all.get(g, {}).get("n", 0)}
            for g in ("B", "L", "D", "A")
        },
        "edit_differences": edit_diffs,
        "repeat_consistency": _dedupe_agreement(by_task, key, repeats),
        "dedupe_conflicts": conflicts,
        "unique_images_scored": len(first_source),
        "protocol_note": "160 originals for effects; 16 repeats for consistency only; "
                         "p03-p08/seed101 primary edit guardrail; person bootstrap 95% descriptive.",
    }
    return summary


def _task_image_id(image_key: str) -> str:
    return image_key.split("|", 1)[0]
