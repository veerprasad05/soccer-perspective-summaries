# Record-grounded summary coverage experiment

The train-only pilots and both 50-match development runs have now completed. Actual results and observed coverage failures are in `notes/record-coverage-results.md`. The run plan below records the original settings; do not proceed to test with these settings before addressing the failed complete-coverage objective.

This branch keeps the approved 200/50/49 match split and pinned Qwen revision. The 49-match test fold has not been generated. Run two prompt modes from the same rendered BBC match facts and audience instruction:

| Mode | System-prompt objective | Output file rule |
|---|---|---|
| `concise` (`record-concise-v1`) | The BBC record is final truth for this task; never invent, infer, or contradict match facts. Aim for at most 150 words. Recorded facts may be selected. | One file per split and mode. |
| `complete` (`record-complete-v1`) | The same truth rule, plus every *available rendered fact*: both teams, competition/date when present, final score, each listed event and its available fields, and all listed statistics. Use compact clauses; coverage takes priority if 150 words is insufficient. | One file per split and mode. |

Both modes use `configs/record_coverage.json`: `max_new_tokens=250`, greedy decoding, seed 42, and the same pinned Qwen2.5-7B-Instruct revision as the baseline. The factual-input hash stays identical between modes for a given match. Prompt version, mode, prompt hash, settings, run ID, output word count, generated token count, and a `hit_max_new_tokens` flag are saved separately. A true cap flag means the output reached 250 tokens and needs inspection; it is not by itself proof of truncation. Generation rejects mixing modes or changed configurations in one output file. Existing baseline files are preserved.

**Length constraint:** 150 words is a target, not proof that every recorded fact can fit. The dev records have a median of 11 events; 33/50 rendered fact inputs exceed 150 words. Compact prose may be shorter, but the 250-token generation cap could still truncate a complete report. Run a train-only GPU pilot and inspect output lengths and SportsSum event coverage before scheduling all development matches. If the cap fails, document a revised limit before the main comparison.

**Evaluation:** `src.sportssum` now reports separate precision, recall, and harmonic-mean F1 for player names, all recorded player-events, and salient goal/red-card events. Omitting a recorded event lowers recall; asserting an event absent from the record lowers precision if the parser recognizes it. The BBC record is the experimental authority. SportsSum-style matching does not assess competition/date, score, event minute/team, or statistics, and cannot establish that the `complete` arm actually included *every* fact. Treat a full-field coverage check as required before making that claim. Compare output lengths and claim precision alongside recall; a longer report has more opportunities to include facts and to make errors.

## Local preflight

```bash
python3 -m unittest discover -s tests -q
python3 -m src.generate --config configs/record_coverage.json --coverage-mode concise --split train --conditions neutral,fan_A,fan_B --smoke-test 2 --output results/preview_concise.jsonl --dry-run
python3 -m src.generate --config configs/record_coverage.json --coverage-mode complete --split train --conditions neutral,fan_A,fan_B --smoke-test 2 --output results/preview_complete.jsonl --dry-run
```

Use `jobs/baselines.sbatch` with environment variables `CONFIG=configs/record_coverage.json`, `COVERAGE_MODE=concise|complete`, `SPLIT`, `CONDITIONS`, and a unique `OUTPUT`. Optional `SMOKE_TEST=N` works only on `train`. After the train-only GPU pilot, run each arm on the 50-match dev split. Freeze prompts and limits before one final run on the test fold.
