# Concise fan-perspective activation and steering experiment

This implements the internal-analysis and steering portions of the submitted proposal, with the TA's emphasis on causal controls. The data source is the `record-concise-v1` prompt family: 200 train matches × paired fan A/B outputs for direction estimation, and the existing 50-match concise neutral dev baseline for steering. The full concise train-pair file has **not yet been generated** locally. The 49-match test fold remains inaccessible to these CLIs.

## Fixed method

- **Encoding context:** every generated fan A/B summary is re-encoded in Qwen under the same neutral system instruction, `Read the following soccer match summary. Do not write a response.`, followed by `Summary:\n` and the summary text. The original supporter generation prompts are absent. This implements the original proposal's neutral-instruction comparison.
- **Representation:** last summary token, no padding, from zero-based decoder layers 12, 20, and 26. Hugging Face `hidden_states[layer+1]` is used for nonfinal layers. The fast tokenizer's character offsets identify the final summary token, excluding chat-template closing tokens. A direction is the mean across matched train pairs of `h(fan_A) - h(fan_B)`. Vectors are saved as CPU float32 safetensors with a separate JSON provenance file. A two-pair smoke vector is marked as such and cannot be used on dev.
- **Causal intervention:** the neutral `record-concise-v1` prompt is regenerated on the same Qwen revision and decoding settings. A temporary forward hook adds `alpha × vector` at the chosen decoder layer's last token during prefill and at the current token during cached decoding. The hook never changes model weights and is removed even on failure. Zero uses no hook and must reproduce the saved neutral baseline exactly.
- **Controls:** positive and negative strengths for the learned vector; equal-norm random directions at the same layer, positions, and strengths, with recorded seeds; no intervention. The sign `A-B` makes positive alpha a *hypothesized* shift toward Team A, pending independent perspective measurement. This adapts CAA to free-form soccer summaries rather than claiming an exact reproduction of its answer-token design.
- **Evaluation:** steering outputs have an `evaluation_condition` for each arm, so `src.sportssum` and `src.evaluate` can score them separately against the BBC record. SportsSum v2 still has limited event extraction and does not measure allegiance. A blinded perspective score and a small check of the scorer's errors are needed before choosing layer/strength or claiming causal success. Do not infer perspective from sentiment alone.

The available concise data holds soccer affinity fixed at `casual`, so the current vector measures the fan A/B contrast only. The TA's emotional-affinity interpolation and good-news/bad-news event probe require separately designed paired inputs; they have not been run. Because the opposing generated summaries differ in wording and selected events, the A/B direction may encode lexical style or team names in addition to supporter perspective.

## Preflight and GPU order

Run from the Insomnia login node in the scratch repository after pulling the branch. The batch scripts require an explicit GPU request at submission and load Qwen only inside the allocated job. The existing `.venv` must have `.[gpu]` installed.

First generate all 400 concise fan A/B training summaries. The earlier two-match pilot is insufficient for a final direction:

```bash
SPLIT=train CONDITIONS=fan_A,fan_B CONFIG=configs/record_coverage.json COVERAGE_MODE=concise OUTPUT=results/train_record_concise_pairs.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL --job-name=concise-train-pairs jobs/baselines.sbatch
```

The existing two-match pilot can exercise activation extraction and hook behavior while that full job is queued. Give each smoke artifact its own path:

```bash
PAIRS=results/train_record_concise_smoke.jsonl VECTORS=results/concise_vectors_smoke.safetensors METADATA=results/concise_vectors_smoke.json SMOKE_TEST=2 LAYERS=12,20,26 sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL --job-name=concise-activation-smoke jobs/activations.sbatch
```

Once the smoke vector exists, exercise no-hook equality and temporary hooks on the same two training matches:

```bash
SPLIT=train SMOKE_TEST=2 VECTORS=results/concise_vectors_smoke.safetensors METADATA=results/concise_vectors_smoke.json BASELINES=results/train_record_concise_smoke.jsonl LAYER=20 STRENGTHS=0.5,1 RANDOM_SEEDS=17 OUTPUT=results/concise_steering_smoke.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL --job-name=concise-steering-smoke jobs/steering.sbatch
```

The steering smoke has 9 arms per match, so expect 18 completed rows. If zero-hook text differs from the saved neutral summary, the job stops with an error and full dev steering must wait. Inspect `results/slurm-JOB_ID.out` and the saved rows; do not use smoke vectors as research evidence.

After the 400-row train-pair file is complete, validate its metadata on the login node without loading Qwen:

```bash
python3 -m src.activations --pairs results/train_record_concise_pairs.jsonl --output results/unused.safetensors --metadata results/unused.json --validate-only
```

Then extract full vectors:

```bash
PAIRS=results/train_record_concise_pairs.jsonl VECTORS=results/concise_vectors_full.safetensors METADATA=results/concise_vectors_full.json LAYERS=12,20,26 sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL --job-name=concise-activations jobs/activations.sbatch
```

After checking the smoke invariants and full-vector metadata, run a development steering grid for a predeclared layer. The example below uses layer 20, strengths 0.5 and 1, and three equal-norm random seeds: 17 arms × 50 matches = 850 summaries. Record the choice before running and use a new output path for each layer/grid. Select layer and strengths on dev; keep test untouched.

```bash
SPLIT=dev VECTORS=results/concise_vectors_full.safetensors METADATA=results/concise_vectors_full.json BASELINES=results/dev_record_concise.jsonl LAYER=20 STRENGTHS=0.5,1 RANDOM_SEEDS=17,29,43 OUTPUT=results/concise_steering_dev_l20.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL --job-name=concise-steering-dev jobs/steering.sbatch
```

Score each arm with the CPU-only commands below after all 850 rows complete:

```bash
python3 -m src.sportssum --generations results/concise_steering_dev_l20.jsonl --split-ids data/splits/dev.json --output results/concise_steering_dev_l20_sportssum.csv --details results/concise_steering_dev_l20_sportssum.jsonl
python3 -m src.evaluate --generations results/concise_steering_dev_l20.jsonl --output results/concise_steering_dev_l20_rules.csv --details results/concise_steering_dev_l20_rules.jsonl
```

The SportsSum CSV measures name/event matching and the rule CSV has narrow score/winner/scorer checks. Neither is a perspective measure or complete semantic fact verification. Use paired comparisons by match when the perspective rubric is available. The TA's proposed good-news/bad-news event probe remains a separate follow-up; it is not silently replaced by these steering jobs.
