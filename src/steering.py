"""Add a train-derived fan direction to Qwen's residual stream on neutral prompts.

Development-only tuning compares no intervention, learned positive/negative
directions, and norm-matched random controls. The test fold is inaccessible.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from .data.common import atomic_jsonl, canonical_json, read_jsonl, sha256_file
from .prompting.render import input_hash
from .prompting.templates import build_messages


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def parse_strengths(value):
    numbers = sorted(set(float(item) for item in value.split(",")))
    if not numbers or any(number <= 0 or not number < float("inf") for number in numbers):
        raise ValueError("Strengths must be positive finite numbers; zero is the no-hook baseline")
    return numbers


def parse_seeds(value):
    seeds = sorted(set(int(item) for item in value.split(",")))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Provide nonnegative random-control seeds")
    return seeds


def experiment_grid(strengths, random_seeds):
    experiments = [("none", 0.0, None)]
    for magnitude in strengths:
        for alpha in (-magnitude, magnitude):
            experiments.append(("learned", alpha, None))
            experiments.extend(("random", alpha, seed) for seed in random_seeds)
    return experiments


def arm_label(control, alpha, seed):
    if control == "none":
        return "none_alpha_0"
    label = f"{control}_alpha_{alpha:+g}"
    return f"{label}_seed_{seed}" if seed is not None else label


def validate_vector_metadata(metadata, config, manifest_hash, layer, split):
    if metadata.get("method") != "paired_mean_last_summary_token_A_minus_B_v1":
        raise ValueError("Vector was extracted with another method")
    if metadata.get("model_id") != config["model_id"] or metadata.get("revision") != config["revision"]:
        raise ValueError("Vector model/revision differs from generation config")
    if metadata.get("split_manifest_sha256") != manifest_hash:
        raise ValueError("Vector was trained with another split manifest")
    if metadata.get("pair_prompt_version") != "record-concise-v1":
        raise ValueError("Steering requires concise fan-pair vectors")
    if metadata.get("direction_sign") != "fan_A minus fan_B" or layer not in metadata.get("layers", []):
        raise ValueError("Vector direction sign or layer is invalid")
    if split == "dev" and (metadata.get("smoke_test") or metadata.get("n_pairs") != 200):
        raise ValueError("Development steering requires all 200 training pairs")


def add_last_token(output, delta):
    """Return a new decoder-layer output with a residual added at its last token."""
    if isinstance(output, tuple):
        hidden, rest = output[0], output[1:]
    elif isinstance(output, list):
        hidden, rest = output[0], output[1:]
    else:
        hidden, rest = output, None
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != delta.numel():
        raise ValueError("Steering vector and decoder output have incompatible shapes")
    changed = hidden.clone()
    changed[:, -1, :] = hidden[:, -1, :] + delta.to(device=hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (changed, *rest)
    if isinstance(output, list):
        return [changed, *rest]
    return changed


def generate_with_direction(generator, messages, settings, layer, delta=None):
    """Register one temporary hook; alpha zero uses no hook at all."""
    if delta is None:
        return generator.generate(messages, settings)
    module = generator.model.model.layers[layer]
    handle = module.register_forward_hook(lambda _module, _inputs, output:
                                          add_last_token(output, delta))
    try:
        return generator.generate(messages, settings)
    finally:
        handle.remove()


def random_direction(vector, seed, torch):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(vector.shape, generator=rng, dtype=torch.float32)
    return sample * (torch.linalg.vector_norm(vector) / torch.linalg.vector_norm(sample))


def validate_baselines(rows, selected_ids, manifest_hash, revision=None, settings=None):
    baselines = {}
    for row in rows:
        if row.get("match_id") not in selected_ids or row.get("condition") != "neutral":
            continue
        if row.get("status") != "complete" or row.get("coverage_mode") != "concise" or row.get("mock"):
            raise ValueError("Baseline must be a completed concise neutral generation")
        if row.get("split_manifest_sha256") != manifest_hash:
            raise ValueError("Baseline split manifest differs")
        if revision is not None and row.get("revision") != revision:
            raise ValueError("Baseline model revision differs")
        if settings is not None and row.get("settings") != settings:
            raise ValueError("Baseline generation settings differ")
        match_id = row["match_id"]
        if match_id in baselines:
            raise ValueError(f"Duplicate neutral baseline for {match_id}")
        baselines[match_id] = row
    if set(baselines) != set(selected_ids):
        raise ValueError("Missing concise neutral baselines for selected matches")
    return baselines


def run(args):
    import torch
    from safetensors.torch import load_file
    from .models.qwen import QwenGenerator

    if args.split not in ("train", "dev") or (args.split == "train") != (args.smoke_test is not None):
        raise ValueError("Use full dev or a train-only --smoke-test; test is reserved")
    if not torch.cuda.is_available():
        raise RuntimeError("Steering requires an allocated CUDA GPU")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    split_dir = Path(config["splits_dir"])
    manifest_hash = sha256_file(split_dir / "manifest.json")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    validate_vector_metadata(metadata, config, manifest_hash, args.layer, args.split)
    if metadata.get("train_ids_sha256") != sha256_file(split_dir / "train.json"):
        raise ValueError("Vector training IDs differ from this repository's train split")
    if sha256_file(args.vectors) != metadata.get("vectors_sha256"):
        raise ValueError("Vector checksum differs from extraction metadata")
    vectors = load_file(args.vectors)
    key = f"layer_{args.layer}"
    if key not in vectors:
        raise ValueError(f"Missing {key} in vector file")
    direction = vectors[key].float()
    if direction.ndim != 1 or not torch.isfinite(direction).all() or torch.linalg.vector_norm(direction) == 0:
        raise ValueError("Steering direction must be a finite nonzero vector")
    ids = json.loads((split_dir / f"{args.split}.json").read_text(encoding="utf-8"))
    if args.smoke_test is not None:
        if not 1 <= args.smoke_test <= len(ids):
            raise ValueError("--smoke-test must select a positive training subset")
        ids = ids[:args.smoke_test]
    selected = set(ids)
    if not selected:
        raise ValueError("No selected matches")
    settings = {key: config[key] for key in ("max_new_tokens", "do_sample", "seed")}
    baselines = validate_baselines(read_jsonl(args.baselines), selected, manifest_hash,
                                   config["revision"], settings)
    records = {}
    for row in read_jsonl(config["matches"]):
        if row["match_id"] in selected:
            if row["match_id"] in records:
                raise ValueError(f"Duplicate source match {row['match_id']}")
            records[row["match_id"]] = row
    if set(records) != selected:
        raise ValueError("Selected matches are missing from processed BBC records")
    strengths = parse_strengths(args.strengths)
    seeds = parse_seeds(args.random_seeds)
    grid = experiment_grid(strengths, seeds)
    experiment_id = digest({"split": args.split, "ids": ids,
                            "vectors_sha256": metadata["vectors_sha256"],
                            "layer": args.layer, "strengths": strengths,
                            "random_seeds": seeds, "settings": settings,
                            "model_id": config["model_id"], "revision": config["revision"],
                            "manifest_sha256": manifest_hash})
    existing = list(read_jsonl(args.output)) if Path(args.output).exists() else []
    if any(row.get("experiment_id") != experiment_id for row in existing):
        raise ValueError("Output contains another steering grid or configuration; use a new file")
    if len({row.get("run_id") for row in existing}) != len(existing):
        raise ValueError("Output contains duplicate run IDs")
    completed = {row["run_id"] for row in existing if row.get("status") == "complete"}
    generator = QwenGenerator(config["model_id"], config["revision"])
    if args.layer >= len(generator.model.model.layers):
        raise ValueError("Selected layer exceeds model depth")
    for index, match_id in enumerate(ids, 1):
        record = records[match_id]
        messages = build_messages(record, "neutral", config["affinity"], "concise")
        prompt_hash = digest(messages)
        baseline = baselines[match_id]
        if baseline.get("prompt_hash") != prompt_hash or baseline.get("input_hash") != input_hash(record):
            raise ValueError(f"Neutral prompt or factual input differs for {match_id}")
        for control, alpha, seed in grid:
            run_id = digest({"match_id": match_id, "control": control, "alpha": alpha,
                             "random_seed": seed, "layer": args.layer,
                             "vectors_sha256": metadata["vectors_sha256"],
                             "prompt_hash": prompt_hash, "settings": settings})
            if run_id in completed:
                continue
            if control == "none":
                delta = None
            else:
                vector = direction if control == "learned" else random_direction(direction, seed, torch)
                delta = alpha * vector
            started = time.monotonic()
            text = generate_with_direction(generator, messages, settings, args.layer, delta)
            if control == "none" and text != baseline["output_text"]:
                raise RuntimeError(f"Zero-hook generation differs from neutral baseline for {match_id}")
            row = {"run_id": run_id, "experiment_id": experiment_id,
                   "match_id": match_id, "split": args.split,
                   "condition": "neutral", "coverage_mode": "concise",
                   "evaluation_condition": arm_label(control, alpha, seed),
                   "prompt_version": "record-concise-v1", "prompt_hash": prompt_hash,
                   "messages": messages,
                   "input_hash": input_hash(record), "split_manifest_sha256": manifest_hash,
                   "model_id": config["model_id"], "revision": config["revision"],
                   "loaded_revision": generator.revision,
                   "gpu_name": torch.cuda.get_device_name(), "precision": "bfloat16",
                   "settings": settings, "layer": args.layer, "control": control,
                   "alpha": alpha, "random_seed": seed,
                   "vectors_sha256": metadata["vectors_sha256"],
                   "output_text": text, "output_word_count": len(text.split()),
                   "output_token_count": generator.last_generated_tokens,
                   "hit_max_new_tokens": generator.last_hit_max_new_tokens,
                   "elapsed_seconds": round(time.monotonic() - started, 3),
                   "status": "complete"}
            existing.append(row)
            atomic_jsonl(args.output, existing)
            completed.add(run_id)
        if index % 10 == 0 or index == len(ids):
            print(f"Completed steering grid for {index}/{len(ids)} matches", flush=True)
    print(json.dumps({"matches": len(ids), "experiments_per_match": len(grid),
                      "completed_rows": len(completed), "output": args.output}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/record_coverage.json")
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--smoke-test", type=int)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--strengths", required=True,
                        help="Comma-separated positive magnitudes; both signs are tested")
    parser.add_argument("--random-seeds", default="17,29,43")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
