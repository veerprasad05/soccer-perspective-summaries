# Development baseline audit — November 18, 2026

## Verified generation

The synced Insomnia outputs contain 400 completed training summaries (200 match-aligned fan A/B pairs) and 150 completed development summaries (50 match-aligned neutral/A/B triples). All rows use the pinned Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`. There are no generation errors, duplicate run IDs, missing split IDs, or within-match factual-input hash mismatches. No test-set generation has been run.

## Factuality: corrected rule-based check, limited coverage

The original `results/dev_baselines_summary_v1_invalid.csv` must not be cited. Its score matcher mistakenly read `Manchester City defeated Crystal Palace 1-0` as `Crystal Palace 1-0`, and sometimes treated shot counts as final scores. Its scorer matcher also treated team names as player names. The evaluator was corrected and regression-tested. Corrected output is `results/dev_baselines_summary.csv`; claim-level details are in `results/dev_evaluation.jsonl`.

| Condition | Dev summaries | Verifiable claims | Summaries with a verifiable claim | Supported / verifiable | Flagged summaries |
|---|---:|---:|---:|---:|---:|
| Neutral | 50 | 91 | 38 | 90 / 91 | 1 / 50 |
| Fan A | 50 | 14 | 13 | 14 / 14 | 0 / 50 |
| Fan B | 50 | 22 | 20 | 22 / 22 | 0 / 50 |

The single detected contradiction is `soccernet-bbc:229`: the neutral output says Metz defeated Paris SG, although the supported final score is Metz 2, Paris SG 3. The high supported-claim fractions are **not** comprehensive factual precision: the parser sees very few fan-summary claims and misses many forms of score, scorer, card, time, and statistical assertion. `event_coverage` in the CSV is a scorer-name mention proxy over 46 development matches with recorded goals, not true event recall. Incomplete source goal logs and shortened scorer names further limit it. The rule checker needs a fixed manual audit before report-level factuality claims.

## Perspective and immediate review

Perspective has not yet been scored; the CSV intentionally says `pending`. A spot check found an unreplaced `Team A` phrase in the fan A output for `soccernet-bbc:338`, and the fan B output for the 5-0 Real Madrid–Granada CF match (`soccernet-bbc:463`) mostly centers the winning Real Madrid side. These examples show why prompts alone cannot be assumed to control allegiance on every match.

`python3 -m src.audit_sheet` generated `results/dev_manual_audit.csv`: 18 summaries from six development matches, two each from A wins, B wins, and draws. The sheet hides condition and contains the shared factual input, summary, and blank stance, emotional-intensity, and factual-notes fields. Rate stance on the project scale **-3 = Team A, 0 = neutral, +3 = Team B** without looking at `results/dev_manual_audit_key.csv`; use the key only after ratings are complete. This is a small manual diagnostic sample, not a population estimate or an inter-annotator reliability study.

## Next implementation milestone

Implement training-only Qwen activation extraction from the 200 fan A/B pairs, with matched prompt positions and layer definitions; then test zero-coefficient and hook-cleanup invariants before development-set steering. `src/activations.py`, `src/steering.py`, and their Slurm jobs are currently stubs. Keep the 49-match test split untouched during method selection.
