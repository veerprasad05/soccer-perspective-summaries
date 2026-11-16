"""Idempotent baseline generation. Atomic whole-file rewrite preserves completed rows."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from .data.common import atomic_jsonl, canonical_json, read_jsonl, sha256_file
from .prompting.render import input_hash
from .prompting.templates import build_messages, CONDITIONS, VERSION

def digest(obj): return hashlib.sha256(canonical_json(obj).encode()).hexdigest()

def run(config, split, conditions, output, generator=None, dry_run=False, smoke_test=None):
    if split not in ("train","dev","test"): raise ValueError("split must be train/dev/test")
    if any(c not in CONDITIONS for c in conditions): raise ValueError("conditions must be neutral,fan_A,fan_B")
    if smoke_test is not None and split != "train": raise ValueError("smoke-test is train-only")
    mock = isinstance(generator, MockGenerator)
    if not dry_run and not mock and config["revision"] == "main":
        raise ValueError("Pin the Qwen revision to a verified commit before real generation")
    split_dir=Path(config["splits_dir"]); manifest_path=split_dir/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    ids=json.loads((split_dir/f"{split}.json").read_text(encoding="utf-8"))
    if len(ids)!=manifest["counts"][split] or len(ids)!=len(set(ids)): raise ValueError("Split count or ID uniqueness mismatch")
    if smoke_test is not None: ids=ids[:smoke_test]
    wanted=set(ids); records={}
    for row in read_jsonl(config["matches"]):
        if row["match_id"] in wanted:
            if row["match_id"] in records: raise ValueError("Duplicate match ID in records")
            records[row["match_id"]]=row
    if len(records)!=len(wanted): raise ValueError(f"Missing {len(wanted)-len(records)} split matches from processed input")
    settings={k:config[k] for k in ("max_new_tokens","do_sample","seed")}
    existing=list(read_jsonl(output)) if Path(output).exists() else []
    completed={r["run_id"] for r in existing if r.get("status")=="complete"}
    new=[]
    for mid in ids:
        rec=records[mid]
        for condition in conditions:
            messages=build_messages(rec,condition,config["affinity"])
            prompt_hash=digest(messages)
            model_id = "mock" if mock else config["model_id"]
            revision = "synthetic" if mock else config["revision"]
            run_id=digest({"match_id":mid,"condition":condition,"model_id":model_id,"revision":revision,"prompt_hash":prompt_hash,"settings":settings})
            if run_id in completed: continue
            base={"run_id":run_id,"match_id":mid,"split":split,"team_a":rec["team_a"],"team_b":rec["team_b"],"condition":condition,
                  "affinity":config["affinity"],"prompt_version":VERSION,"messages":messages,"prompt_hash":prompt_hash,"input_hash":input_hash(rec),
                  "model_id":model_id,"revision":revision,"mock":mock,"settings":settings,
                  "seed":settings["seed"],"split_manifest_sha256":sha256_file(manifest_path)}
            if dry_run:
                new.append({**base,"status":"dry_run","output_text":None,"elapsed_seconds":0}); continue
            start=time.monotonic()
            try:
                text=generator.generate(messages,settings)
                row={**base,"status":"complete","output_text":text,"error":None,"elapsed_seconds":round(time.monotonic()-start,3)}
            except Exception as exc:
                row={**base,"status":"error","output_text":None,"error":f"{type(exc).__name__}: {exc}","elapsed_seconds":round(time.monotonic()-start,3)}
            existing=[e for e in existing if e.get("run_id")!=run_id]
            existing.append(row); atomic_jsonl(output,existing)
            new.append(row)
    if dry_run: print(json.dumps(new[:3],indent=2)); return new
    expected=len(ids)*len(conditions)
    counts={c:sum(r.get("status")=="complete" and r.get("split")==split and r.get("condition")==c and r.get("match_id") in wanted for r in existing) for c in conditions}
    print(json.dumps({"expected":expected,"completed_by_condition":counts,"errors":sum(r.get("status")=="error" for r in new)},indent=2))
    return new

class MockGenerator:
    def generate(self,messages,settings): return "Mock output for pipeline testing only."

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/baselines.json");p.add_argument("--split",required=True);p.add_argument("--conditions",default="neutral,fan_A,fan_B");p.add_argument("--output",required=True);p.add_argument("--dry-run",action="store_true");p.add_argument("--mock",action="store_true");p.add_argument("--smoke-test",type=int)
    a=p.parse_args();config=json.loads(Path(a.config).read_text(encoding="utf-8"))
    generator=None if a.dry_run else MockGenerator() if a.mock else __import__("src.models.qwen",fromlist=["QwenGenerator"]).QwenGenerator(config["model_id"],config["revision"])
    run(config,a.split,a.conditions.split(","),a.output,generator,a.dry_run,a.smoke_test)
if __name__=="__main__":main()
