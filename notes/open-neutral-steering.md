# Open neutral prompt steering check

This is a new, versioned neutral prompt mode for the requested A6000 smoke
comparison. The user message begins exactly:

> Write a summary for the match.

It then supplies the same `Verified match facts:` block from the BBC record.
The system message is **identical** to `record-concise-v1`: the BBC record is
final truth, invented or contradictory match facts are forbidden, and the
target is at most 150 words. The `casual` affinity no longer adds "Use
restrained language" in this mode. The new version is
`record-concise-open-v1` (`COVERAGE_MODE=concise_open`). It is reserved for
neutral generation; the 200-pair A-minus-B vector still comes from the
existing concise fan summaries. The test split remains unused.

The existing `train_record_concise_neutral_a6000.jsonl` uses the **old**
prompt and cannot be supplied as the baseline for this experiment. Generate
a fresh two-match neutral baseline on A6000, using a new path:

```bash
SPLIT=train CONDITIONS=neutral CONFIG=configs/record_coverage.json COVERAGE_MODE=concise_open SMOKE_TEST=2 OUTPUT=results/train_record_concise_open_neutral_a6000.jsonl sbatch -A edu -p short --gres=gpu:A6000:1 --time=01:00:00 --export=ALL --job-name=open-neutral-a6000 jobs/baselines.sbatch
```

After this job finishes, expect two completed rows. Then run all requested
strengths on the same two training matches, with the no-hook and one matched
random control per sign and strength:

```bash
SPLIT=train SMOKE_TEST=2 NEUTRAL_PROMPT_MODE=concise_open VECTORS=results/concise_vectors_full.safetensors METADATA=results/concise_vectors_full.json BASELINES=results/train_record_concise_open_neutral_a6000.jsonl LAYER=20 STRENGTHS=0.5,1,2,5,10 RANDOM_SEEDS=17 OUTPUT=results/concise_open_steering_smoke_a6000_all_alphas.jsonl sbatch -A edu -p short --gres=gpu:A6000:1 --time=01:00:00 --export=ALL --job-name=open-steering-a6000 jobs/steering.sbatch
```

The complete smoke file should have **42 rows** (21 arms per match). If the
no-hook exact-text check fails again, keep the partial file and log for
diagnosis; do not silently mix old and new baselines or GPUs in one output.
Compare changes against the no-hook and random arms, and inspect the score,
winner, scorers, and event claims against the BBC record before a dev run.
Larger alpha values can amplify an intervention but may also harm factuality
or readability.

## Two-match A6000 smoke result (2026-11-18)

The synced baseline has 2 complete rows. The steering output has 42 complete
rows: 2 no-hook, 20 learned, and 20 random-control generations. Both no-hook
texts match their baseline exactly. All rows use the pinned model revision,
layer 20, `record-concise-open-v1`, and the A6000. None hit the 250-token cap;
outputs contain 57–125 words.

The smoke is **not a factuality pass**. BBC record `soccernet-bbc:433` lists
five Real Madrid scorers, including Cristiano Ronaldo at 90+2, and four
Getafe yellow cards plus one Real Madrid yellow card. The no-hook summary
names only four Real Madrid scorers and says the card counts were four and
three. BBC record `soccernet-bbc:075` gives Leicester five corners and West
Brom six; the no-hook summary says five each. Learned alpha -10 for match 433
says Getafe scored twice and gives several event details incorrectly. Learned
alpha +10 reverses the card counts. Random controls at large alpha also contain
incorrect event details, so these examples do not isolate a learned-vector
effect.

The automated SportsSum-style and rule evaluations were run on all 42 rows;
their CSV and JSONL outputs are alongside the generations in `results/`.
SportsSum-style precision and rule fact precision are 1.0 for every arm,
including the erroneous no-hook outputs. These scores cover extracted claims
and miss several wrong counts, event minutes, and omissions visible in the
record comparison. Event recall varies by arm, but only two training matches
and one random seed are present; it is not an estimate of a dev-set effect.

Do not infer successful A/B stance steering or BBC-grounded factuality from
this smoke. The project owner elected to run the full dev sweep with these
known limitations. Interpret its factuality scores alongside record-level
checks, and keep the test split untouched.
