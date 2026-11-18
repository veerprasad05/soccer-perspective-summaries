"""Train-only Qwen activations from record-grounded concise fan pairs.

Under one neutral analysis chat, the saved direction is the mean paired
last-summary-token residual difference h(fan_A) - h(fan_B).
"""

import argparse
import json
import subprocess
from pathlib import Path

from .data.common import atomic_json, read_jsonl, sha256_file

ANALYSIS_SYSTEM = "Read the following soccer match summary. Do not write a response."
ANALYSIS_PREFIX = "Summary:\n"
PAIR_PROMPT_VERSION = "record-concise-v1"


def validate_train_pairs(rows, expected_ids, revision, manifest_hash):
    """Return complete, aligned fan pairs; reject leakage and stale prompts."""
    expected_ids = set(expected_ids)
    pairs = {}
    for row in rows:
        match_id = row.get("match_id")
        condition = row.get("condition")
        if match_id not in expected_ids or row.get("split") != "train":
            raise ValueError(f"Non-training or unexpected match in activation pairs: {match_id}")
        if condition not in ("fan_A", "fan_B") or row.get("status") != "complete":
            raise ValueError("Need completed fan_A and fan_B rows only")
        if row.get("coverage_mode") != "concise" or row.get("prompt_version") != PAIR_PROMPT_VERSION:
            raise ValueError("Activation pairs must use record-concise-v1 prompts")
        if row.get("revision") != revision or row.get("split_manifest_sha256") != manifest_hash:
            raise ValueError("Model revision or split manifest differs from the configured run")
        if row.get("mock") or not row.get("output_text", "").strip():
            raise ValueError("Activation pairs require nonempty real model outputs")
        by_condition = pairs.setdefault(match_id, {})
        if condition in by_condition:
            raise ValueError(f"Duplicate {condition} generation for {match_id}")
        by_condition[condition] = row
    if set(pairs) != expected_ids:
        raise ValueError(f"Missing {len(expected_ids - set(pairs))} training matches")
    for match_id, pair in pairs.items():
        if set(pair) != {"fan_A", "fan_B"}:
            raise ValueError(f"Incomplete A/B pair for {match_id}")
        a, b = pair["fan_A"], pair["fan_B"]
        if (not a.get("input_hash") or a.get("input_hash") != b.get("input_hash") or
                a.get("affinity") != b.get("affinity")):
            raise ValueError(f"Mismatched factual input or affinity for {match_id}")
        if a.get("settings") != b.get("settings"):
            raise ValueError(f"Mismatched generation settings for {match_id}")
    return pairs


def summary_positions(offsets, start, end):
    """Token indices overlapping summary text, excluding chat-template tokens."""
    if not 0 <= start < end:
        raise ValueError("Summary span must be nonempty")
    indices = [i for i, (left, right) in enumerate(offsets)
               if right > left and right > start and left < end]
    if not indices:
        raise ValueError("Tokenizer did not expose summary token positions")
    return indices


def analysis_inputs(tokenizer, summary):
    messages = [{"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user", "content": ANALYSIS_PREFIX + summary}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False)
    start = rendered.rfind(summary)
    if start < 0:
        raise ValueError("Rendered analysis chat lost the summary")
    encoded = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    positions = summary_positions(offsets, start, start + len(summary))
    return encoded, positions[-1]


def extract_one(model, tokenizer, summary, layers, torch):
    encoded, position = analysis_inputs(tokenizer, summary)
    with torch.inference_mode():
        output = model.model(**encoded.to(model.device), output_hidden_states=True,
                             use_cache=False, return_dict=True)
    # hidden_states[0] is the embedding; layer l uses hidden_states[l+1].
    result = {layer: output.hidden_states[layer + 1][0, position].detach().float().cpu()
              for layer in layers}
    del output
    return result


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    split_dir = Path(config["splits_dir"])
    train_ids = json.loads((split_dir / "train.json").read_text(encoding="utf-8"))
    if len(train_ids) != 200 or len(set(train_ids)) != 200:
        raise ValueError("The approved training split must contain 200 unique matches")
    if args.smoke_test is not None and not 1 <= args.smoke_test <= 200:
        raise ValueError("--smoke-test must select 1 to 200 training matches")
    selected = train_ids[:args.smoke_test] if args.smoke_test is not None else train_ids
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one distinct training match")
    rows = [row for row in read_jsonl(args.pairs)
            if args.smoke_test is None or
            (row.get("match_id") in selected and row.get("condition") in ("fan_A", "fan_B"))]
    manifest_path = split_dir / "manifest.json"
    pairs = validate_train_pairs(rows, selected, config["revision"], sha256_file(manifest_path))
    expected_settings = {key: config[key] for key in ("max_new_tokens", "do_sample", "seed")}
    for pair in pairs.values():
        for row in pair.values():
            if (row.get("model_id") != config["model_id"] or
                    row.get("affinity") != config["affinity"] or
                    row.get("settings") != expected_settings):
                raise ValueError("Training pair model, affinity, or settings differ from config")
    layers = sorted(set(int(value) for value in args.layers.split(",")))
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("Provide nonnegative zero-based decoder layer indices")
    if args.validate_only:
        print(json.dumps({"validated_training_pairs": len(pairs),
                          "prompt_version": PAIR_PROMPT_VERSION,
                          "layers": layers, "smoke_test": args.smoke_test is not None}, indent=2))
        return
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Activation extraction requires an allocated CUDA GPU")
    output_path, metadata_path = Path(args.output), Path(args.metadata)
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Use new output and metadata paths; vector files are immutable")
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"])
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer with character offsets is required")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["revision"], torch_dtype=torch.bfloat16,
        device_map="auto")
    model.eval()
    if max(layers) >= len(model.model.layers) - 1:
        raise ValueError("Select nonfinal decoder layers; the last hidden state is post-norm")
    sums = {}
    for index, match_id in enumerate(selected, 1):
        pair = pairs[match_id]
        a = extract_one(model, tokenizer, pair["fan_A"]["output_text"], layers, torch)
        b = extract_one(model, tokenizer, pair["fan_B"]["output_text"], layers, torch)
        for layer in layers:
            difference = a[layer] - b[layer]
            sums[layer] = sums.get(layer, torch.zeros_like(difference)) + difference
        if index % 20 == 0 or index == len(selected):
            print(f"Encoded {index}/{len(selected)} matched training pairs", flush=True)
    vectors = {f"layer_{layer}": (sums[layer] / len(selected)).contiguous()
               for layer in layers}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(vectors, str(temp_path))
    temp_path.replace(output_path)
    metadata = {
        "method": "paired_mean_last_summary_token_A_minus_B_v1",
        "analysis_system": ANALYSIS_SYSTEM, "analysis_prefix": ANALYSIS_PREFIX,
        "pooling": "last summary token, no padding",
        "layer_numbering": "zero-based decoder blocks; hidden_states[layer+1]",
        "layers": layers,
        "vector_norms": {str(layer): float(torch.linalg.vector_norm(vectors[f"layer_{layer}"]))
                         for layer in layers},
        "n_pairs": len(selected), "smoke_test": args.smoke_test is not None,
        "direction_sign": "fan_A minus fan_B",
        "pair_prompt_version": PAIR_PROMPT_VERSION,
        "model_id": config["model_id"], "revision": config["revision"],
        "loaded_revision": model.config._commit_hash or config["revision"],
        "pairs_sha256": sha256_file(args.pairs),
        "train_ids_sha256": sha256_file(split_dir / "train.json"),
        "split_manifest_sha256": sha256_file(manifest_path),
        "vectors_sha256": sha256_file(output_path),
        "git_commit": git_commit(), "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(), "precision": "bfloat16",
        "transformers_version": __import__("transformers").__version__,
    }
    atomic_json(metadata_path, metadata)
    print(json.dumps({"vectors": str(output_path), "metadata": str(metadata_path),
                      "n_pairs": len(selected), "norms": metadata["vector_norms"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/record_coverage.json")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--layers", default="12,20,26")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--smoke-test", type=int)
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
