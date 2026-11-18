# Soccer perspective summaries: data and baseline infrastructure

The original proposal targeted 500 matches split 300/100/100. After the archive audit, the user approved **all 299 eligible BBC matches split 200 training / 50 development / 49 test**. The supplied ZIP has 340 raw matches. The revised official split has been created and verified. Qwen generated 400 complete training summaries and 150 complete development summaries on Insomnia; see `notes/sportssum-evaluation.md` for the SportsSum-style development evaluation. The original proposal, graded feedback, midterm template, and research specification remain in the parent directory.

## Local setup

Python 3.10+ is required; this workspace has Python 3.12.3. The data/split/prompt/evaluation code uses the standard library.

```bash
cd soccer-perspective-summaries
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

For a GPU allocation, install `python3 -m pip install -e '.[gpu]'` **inside the allocation/environment** after confirming compatible CUDA/PyTorch builds. Do not load Qwen on a login node. Record exact installed versions with `python3 -m pip freeze` in the run log. `configs/baselines.json` pins Qwen to commit `a09a35458c702b33eeacc393d103063234e8bc28`.

## Dataset workflow

```bash
python3 -m src.data.download --output data/raw/SoccerNet-BBC.zip
python3 -m src.data.inspect_archive --input data/raw/SoccerNet-BBC.zip --output notes/archive-inspection.json
```

The supplied ZIP was verified against the repository Git blob SHA. The downloader remains available for reproduction. The inspector lists every path in its JSON report and prints file types, hierarchy, sizes, and samples. The source-specific ZIP adapter reads match folders, scores, commentary, and statistics; it records conflicts and leaves unknown facts null. No BBC article text is used as generation input.

```bash
python3 -m src.data.normalize --source data/raw/SoccerNet-BBC.zip --output data/processed/matches.jsonl
python3 -m src.data.validate --input data/processed/matches.jsonl --output data/processed/audit.json
python3 -m src.data.split --input data/processed/matches.jsonl --output data/splits --train 200 --dev 50 --test 49 --seed 42
python3 -m src.data.split --input data/processed/matches.jsonl --output data/splits --smoke-test 2
```

The approved split uses every one of the 299 eligible match IDs exactly once. The saved manifest is `data/splits/manifest.json`; its SHA-256 is `2a4aa142931c986c1695b8e40e4fc715ded4d49ecfbf6ba4de2a2ced442ef1ab`. The split rejects fewer than 299 eligible distinct matches and refuses to replace a different manifest. It sorts by match ID, shuffles with `random.Random(42)`, and assigns 200/50/49. A smoke test reads only train IDs and never edits the manifest. The official test IDs must not be used to tune prompts, layers, or coefficients.

## Prompt preview, generation, evaluation

```bash
python3 -m src.generate --config configs/baselines.json --split train --conditions neutral,fan_A,fan_B --output results/train_preview.jsonl --dry-run --smoke-test 2
python3 -m src.generate --config configs/baselines.json --split train --conditions fan_A,fan_B --output results/train_pairs.jsonl
python3 -m src.generate --config configs/baselines.json --split dev --conditions neutral,fan_A,fan_B --output results/dev_baselines.jsonl
python3 -m src.generate --config configs/baselines.json --split test --conditions neutral,fan_A,fan_B --output results/test_baselines.jsonl
python3 -m src.evaluate --matches data/processed/matches.jsonl --generations results/dev_baselines.jsonl --details results/dev_evaluation.jsonl --output results/dev_baselines_summary.csv
python3 -m src.sportssum --matches data/processed/matches.jsonl --generations results/dev_baselines.jsonl --split-ids data/splits/dev.json --output results/sportssum_dev.csv --details results/sportssum_dev_details.jsonl
python3 -m src.audit_sheet --matches data/processed/matches.jsonl --generations results/dev_baselines.jsonl --dev-ids data/splits/dev.json
```

`--mock` exercises the output pipeline with an explicitly labeled fake text and must never be treated as research generation. Each real row records prompt, input hash, split manifest hash, settings, seed, model, revision, elapsed time, completion state, and error. Completed run IDs are skipped; errors are retried. The cluster smoke, training, and development runs completed. `src.sportssum` computes adapted SportsSum name and player-event set F1 against the structured BBC event records; see `notes/sportssum-evaluation.md` for definitions, measured results, and limitations. The separate `src.evaluate` command is a **limited regex checker** for explicit scores, winners, scorer names, and some goal minutes. It marks ambiguous or unrecorded claims `unverifiable`; its precision is only over the claims it recognizes. Neither method alone supports a comprehensive factuality estimate. Perspective remains `pending`. The audit-sheet command creates a condition-blinded 18-summary development review sheet and a separate answer key for later use.

See `notes/hpc-setup.md` for cluster commands. `src/probe.py` remains future work.

## Record coverage prompt experiment

The `experiment/record-coverage` branch adds two BBC-record-grounded prompt modes, `concise` and `complete`, with a 150-word target and `max_new_tokens=250`. The modes share the exact same factual input and are saved to different JSONL files. `complete` asks for every available rendered fact; this is a prompt objective, not an automatically verified outcome. The SportsSum-style evaluator now prints separate precision, recall, and F1, but its event/name parser does not measure all score, time, or statistic fields. See `notes/record-coverage-experiment.md` for the design, limits, preflight commands, and cluster settings. The saved 200/50/49 split remains unchanged; test generation waits until dev settings are frozen.

## Concise activation and steering experiment

The `experiment/concise-steering` branch implements train-only Qwen activation extraction from the record-grounded concise fan A/B summaries and neutral-prompt residual steering with no-hook and norm-matched random controls. CPU tests and validation of the existing two-match concise pilot pass; a GPU extraction/hook smoke run and the full 400-summary concise train-pair generation are still required. The 49-match test fold is blocked from the extraction and steering CLIs. See `notes/concise-steering.md` for exact methods, run order, Slurm commands, and current limits.

## Version control and transfer

This directory is a Git repository. It tracks code, configuration, notes, and the approved match-ID split. The BBC ZIP, parsed match records, archive samples, caches, and generated results are ignored. Use a **private remote** for ongoing code changes, then clone or pull on Insomnia. Transfer `data/processed/matches.jsonl` separately with `rsync`, because Git deliberately omits it; see `notes/hpc-setup.md`. A local `git init` alone does not create a remote that the cluster can pull from.
