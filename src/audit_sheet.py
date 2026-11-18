"""Create a small, condition-blinded development-set review sheet."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .data.common import read_jsonl
from .prompting.render import render_match_facts

CONDITIONS = {"neutral", "fan_A", "fan_B"}


def build_rows(matches, generations, dev_ids, *, seed=42, per_outcome=2):
    records = {r["match_id"]: r for r in matches if r["match_id"] in dev_ids}
    if set(records) != dev_ids:
        raise ValueError("Missing development matches")
    outputs = defaultdict(dict)
    for row in generations:
        mid = row["match_id"]
        if mid not in dev_ids or row.get("status") != "complete":
            continue
        condition = row["condition"]
        if condition not in CONDITIONS or condition in outputs[mid]:
            raise ValueError(f"Duplicate or unknown condition for {mid}")
        outputs[mid][condition] = row
    if any(set(outputs[mid]) != CONDITIONS for mid in dev_ids):
        raise ValueError("Each development match needs all three completed baselines")

    buckets = defaultdict(list)
    for mid, rec in records.items():
        score = rec["score"]
        outcome = "A_win" if score["A"] > score["B"] else "B_win" if score["B"] > score["A"] else "draw"
        buckets[outcome].append(mid)
    rng = random.Random(seed)
    selected = []
    for outcome in ("A_win", "B_win", "draw"):
        mids = sorted(buckets[outcome])
        if len(mids) < per_outcome:
            raise ValueError(f"Only {len(mids)} development matches with outcome {outcome}")
        selected.extend(rng.sample(mids, per_outcome))
    rng.shuffle(selected)

    blinded, key = [], []
    for mid in selected:
        conditions = sorted(CONDITIONS)
        rng.shuffle(conditions)
        for condition in conditions:
            row = outputs[mid][condition]
            audit_id = f"dev-audit-{len(blinded)+1:03d}"
            blinded.append({
                "audit_id": audit_id, "match_id": mid,
                "team_a": records[mid]["team_a"], "team_b": records[mid]["team_b"],
                "factual_input": render_match_facts(records[mid]),
                "summary": row["output_text"],
                "perspective_minus3_to_plus3": "", "emotion_0_to_3": "", "factual_notes": "",
            })
            key.append({"audit_id": audit_id, "match_id": mid, "condition": condition, "run_id": row["run_id"]})
    return blinded, key


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", default="data/processed/matches.jsonl")
    parser.add_argument("--generations", default="results/dev_baselines.jsonl")
    parser.add_argument("--dev-ids", default="data/splits/dev.json")
    parser.add_argument("--output", default="results/dev_manual_audit.csv")
    parser.add_argument("--key", default="results/dev_manual_audit_key.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-outcome", type=int, default=2)
    args = parser.parse_args()
    dev_ids = set(json.loads(Path(args.dev_ids).read_text(encoding="utf-8")))
    blinded, key = build_rows(read_jsonl(args.matches), read_jsonl(args.generations), dev_ids,
                              seed=args.seed, per_outcome=args.per_outcome)
    write_csv(args.output, blinded)
    write_csv(args.key, key)
    print(f"Wrote {len(blinded)} blinded summaries from {len(blinded)//3} development matches to {args.output}")


if __name__ == "__main__":
    main()
