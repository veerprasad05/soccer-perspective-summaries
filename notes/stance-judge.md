# Blinded supporter-perspective judge

`src.stance_judge` rates the **50-match development split only**. It reads the
completed 850-row concise steering JSONL and sends one BBC record and one
candidate summary per request to the OpenAI Responses API. The request omits
the run ID, steering method, coefficient, random seed, and generation prompt.
The output preserves those labels for paired analysis after scoring. The
`bbc-supporter-stance-v1` rubric scores allegiance from -3 (strong Team A) to
+3 (strong Team B), with 0 for neutral or ambiguous summaries, and emotional
intensity separately from 0 to 3. It explicitly says that winning, losing,
scorelines, and fact counts alone are not evidence of allegiance. Factuality
remains a separate evaluation.

`gpt-5.6-luna` is the default because it is an independent, low-cost judge.
OpenAI describes Luna as a cost-sensitive model, not its strongest frontier
tier. Keep `--model` configurable and choose a judge after comparing a fixed
pilot against blinded human ratings. The official model page currently lists
only the `gpt-5.6-luna` alias, so record the API-returned model ID and rubric
version; the code refuses to mix returned model IDs in one output file.

## Run from the local repository

No GPU or extra Python package is required. A live run requires an OpenAI API
key in `OPENAI_API_KEY` and access to the chosen API model. Do not commit the
key. The API receives BBC facts and summary text. A dry run validates the
source and prints only the planned request counts:

```bash
python3 -m src.stance_judge run --output results/stance_judge_pilot_luna.jsonl --pilot-matches 4 --arms none_alpha_0,learned_alpha_-1,learned_alpha_+1,random_alpha_-1_seed_17,random_alpha_+1_seed_17 --dry-run
```

Once the key is available, omit `--dry-run` to rate the same **20** summaries.
This selects four dev matches deterministically with seed 42 and all five
specified arms for each match. Identical match-summary requests are reused
without a second API call, but every source arm receives a separate row.
The output is written atomically after each rating. Repeating the exact
command resumes incomplete work. Use a new output path for a different model,
rubric, or source text, and keep pilot and full ratings separate for clarity.

```bash
python3 -m src.stance_judge run --output results/stance_judge_pilot_luna.jsonl --pilot-matches 4 --arms none_alpha_0,learned_alpha_-1,learned_alpha_+1,random_alpha_-1_seed_17,random_alpha_+1_seed_17
python3 -m src.stance_judge report --judgments results/stance_judge_pilot_luna.jsonl --output results/stance_judge_pilot_luna.csv
```

Make a **blinded** human calibration sheet from that pilot. Fill the blank
human stance and emotion columns without opening the answer key; the key has
both the hidden arm and the LLM ratings. The sample is outcome-aware where
the selected matches allow it. One human rater can audit the judge, but does
not provide inter-rater reliability. Two independent raters are preferable.

```bash
python3 -m src.stance_judge audit --judgments results/stance_judge_pilot_luna.jsonl --matches-count 4 --output results/stance_human_pilot.csv --key results/stance_human_pilot_key.csv
python3 -m src.stance_judge compare-human --sheet results/stance_human_pilot.csv --key results/stance_human_pilot_key.csv
```

The comparison prints exact agreement, agreement within one scale point,
mean absolute error, and whether both raters chose Team A, neutral, or Team B.
These describe this small sample; they are not a proof of judge validity.
If Luna disagrees materially with the human audit, run the same pilot with
`--model gpt-5.6-terra` and a distinct output path, then compare again before
using a judge on all 850 outputs. The chosen judge and rubric should be fixed
on development data before any final test evaluation.

After calibration, rate all dev arms:

```bash
python3 -m src.stance_judge run --output results/stance_judge_dev_l20_luna.jsonl --model gpt-5.6-luna
python3 -m src.stance_judge report --judgments results/stance_judge_dev_l20_luna.jsonl --output results/stance_judge_dev_l20_luna.csv
```

The report gives mean stance, paired mean change from the same match's
no-hook score, mean shift in the intended direction, mean absolute stance,
emotional intensity, and uncertainty by arm. A positive learned A-minus-B
coefficient should move the -3-to-+3 score negative if the intended direction
holds. Compare that movement with the
equal-norm random controls. Keep SportsSum and narrow rule scores beside these
perspective ratings; the LLM stance judge does not certify factual accuracy.

The API integration has local fake-client tests and a real-data dry run. It
has **not** made a live request in this workspace, so API access, returned
schema behavior, and human agreement remain to be verified by the pilot.
