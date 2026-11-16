"""Frozen match-level split; current approved design is 200/50/49."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from .common import read_jsonl, atomic_json, canonical_json, outcome, sha256_file
from .validate import audit

def create_split(rows, output, train=200, dev=50, test=49, seed=42, source_hash=None):
    eligible, report = audit(rows)
    required = train + dev + test
    if len(eligible) < required:
        raise ValueError(f"Need {required} distinct eligible matches ({train}/{dev}/{test}); found {len(eligible)}, shortage {required-len(eligible)}. Exclusion causes: {report['exclusion_reasons']}")
    ordered = sorted(eligible, key=lambda r: r["match_id"])
    rng = random.Random(seed); rng.shuffle(ordered)
    selected = ordered[:required]
    folds = {"train": selected[:train], "dev": selected[train:train+dev], "test": selected[train+dev:]}
    all_ids = [r["match_id"] for rs in folds.values() for r in rs]
    if len(all_ids) != len(set(all_ids)) or [len(folds[k]) for k in ("train","dev","test")] != [train,dev,test]:
        raise AssertionError("Split overlap or wrong fold sizes")
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    diagnostics = {}
    for name, rs in folds.items():
        teams = collections.Counter(t for r in rs for t in (r["team_a"], r["team_b"]))
        diagnostics[name] = {"outcomes": dict(collections.Counter(outcome(r) for r in rs)), "repeated_teams": {k:v for k,v in teams.items() if v>1}}
    manifest = {"seed": seed, "counts": {k:len(v) for k,v in folds.items()}, "target": {"train":train,"dev":dev,"test":test},
                "selection": "sort eligible records by match_id, Python random.Random(seed).shuffle, take first required, assign train/dev/test in order",
                "id_sha256": hashlib.sha256(canonical_json(all_ids).encode()).hexdigest(), "source_sha256": source_hash,
                "eligible_count":len(eligible), "eligible_ids_sha256":hashlib.sha256(canonical_json(sorted(r["match_id"] for r in eligible)).encode()).hexdigest(),
                "excluded_count":report["excluded"], "exclusion_reasons":report["exclusion_reasons"], "diagnostics":diagnostics}
    existing = out / "manifest.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Existing manifest differs: {existing}; choose a new output directory rather than replacing a frozen split")
    for name, rs in folds.items(): atomic_json(out / f"{name}.json", [r["match_id"] for r in rs])
    atomic_json(existing, manifest)
    return manifest

def smoke_ids(split_dir, n):
    ids = json.loads((Path(split_dir)/"train.json").read_text())
    if n < 0 or n > len(ids): raise ValueError("smoke-test N must fit inside train")
    return ids[:n]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", default="data/splits")
    p.add_argument("--train",type=int,default=200);p.add_argument("--dev",type=int,default=50);p.add_argument("--test",type=int,default=49);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--smoke-test",type=int)
    a=p.parse_args()
    if a.smoke_test is not None:
        print(json.dumps({"smoke_train_ids":smoke_ids(a.output,a.smoke_test)},indent=2)); return
    print(json.dumps(create_split(read_jsonl(a.input),a.output,a.train,a.dev,a.test,a.seed,sha256_file(a.input)),indent=2))
if __name__ == "__main__": main()
