# Insomnia setup for vsp2117

The user's setup email specifies `/insomnia001/depts/edu/users/vsp2117` as scratch. Their live `sacctmgr` output shows Slurm account `edu`; `sinfo` advertises `gpu:l40`, `gpu:R6000`, `gpu:h100`, `gpu:A6000`, and `gpu:l40s` on `short` and `burst`. Columbia's [queue guide](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/1169162241/New+Queue+System) says `edu` can use `short` (maximum 12 hours) but not `hpc_test`; `burst` can be preempted. Prefer `short` for the first run. The live scheduler must still accept a job request before GPU access is considered verified.

## 1. Quick GPU allocation check on Insomnia

From the login node, run one short compute job:

```bash
srun -A edu -p short --gres=gpu:l40:1 --cpus-per-task=2 --mem=16G -t 00:05:00 nvidia-smi
```

This may wait in the queue. Record the full output or error. The `gpu:l40:1` syntax requests one L40 GPU, matching the advertised `sinfo` GRES; it does not assume an L40 is idle now. If it is rejected, inspect the exact error before changing resources.

## 2. Transfer the project from the local workspace

The scratch directory is confirmed by the email; create the project directory on the cluster:

```bash
mkdir -p /insomnia001/depts/edu/users/vsp2117/soccer-perspective-summaries
```

From the local machine containing `/home/veerprasad/nlp-proj`, while connected through Columbia VPN when necessary:

```bash
cd /home/veerprasad/nlp-proj
rsync -av --exclude '.venv' --exclude '.hf-cache' --exclude 'data/raw' --exclude 'results' soccer-perspective-summaries/ vsp2117@insomnia.rcs.columbia.edu:/insomnia001/depts/edu/users/vsp2117/soccer-perspective-summaries/
```

This transfers source code, the private processed facts and audit, and the approved 200/50/49 split; it omits the raw BBC ZIP and local caches. On Insomnia, confirm files and checksum:

For ongoing code changes, a private Git remote can replace the **code** portion of this transfer. `git clone PRIVATE_REPO_URL` on Insomnia still needs a separate `rsync` of `data/processed/matches.jsonl`, since the processed source material is intentionally Git-ignored. Do not place a personal access token in a clone command or script.

```bash
cd /insomnia001/depts/edu/users/vsp2117/soccer-perspective-summaries
ls -l data/processed/matches.jsonl data/splits/manifest.json
sha256sum data/splits/manifest.json
mkdir -p results
```

Expected manifest SHA-256: `2a4aa142931c986c1695b8e40e4fc715ded4d49ecfbf6ba4de2a2ced442ef1ab`.

## 3. Python environment and model smoke run

Use an allocated compute node for package installation and model loading. `configs/baselines.json` pins Qwen to commit `a09a35458c702b33eeacc393d103063234e8bc28` from the [Hugging Face model API](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct). The smoke script reads that same config. The jobs put the Hugging Face cache under the scratch project rather than the smaller home directory.

```bash
srun --pty -A edu -p short --gres=gpu:l40:1 --cpus-per-task=4 --mem=32G -t 01:00:00 /bin/bash
cd /insomnia001/depts/edu/users/vsp2117/soccer-perspective-summaries
python3 --version
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[gpu]'
export HF_HOME="$PWD/.hf-cache"
python3 -m src.models.gpu_smoke --load-model
python3 -m pip freeze > results/pip-freeze.txt
```

If the cluster's Python/CUDA environment needs a different PyTorch wheel, choose the correct wheel from the installed CUDA version before installing the project; record the versions. A successful `nvidia-smi` check does not prove the model can load. Do not run Qwen on the login node.

## 4. Baselines after the model smoke succeeds

On Insomnia, from the project directory, run a train-only two-match smoke run before the full baseline. The batch script now also accepts `SMOKE_TEST=2` for a noninteractive train-only job:

```bash
source .venv/bin/activate
export HF_HOME="$PWD/.hf-cache"
python3 -m src.generate --config configs/baselines.json --split train --conditions neutral,fan_A,fan_B --smoke-test 2 --output results/train_smoke.jsonl
```

Then use the batch script for the 200 training contrast pairs and 50 development baselines:

```bash
mkdir -p results
SPLIT=train CONDITIONS=fan_A,fan_B OUTPUT=results/train_pairs.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL jobs/baselines.sbatch
SPLIT=dev CONDITIONS=neutral,fan_A,fan_B OUTPUT=results/dev_baselines.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL jobs/baselines.sbatch
```

These are explicit user-submitted jobs; no job is submitted by the repository. Monitor with `squeue -u "$USER"`, `sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS`, and `results/slurm-JOB_ID.out`. Repeating the same generation command resumes completed rows. Run the test fold only after methods and prompts are fixed on development data.

Columbia's [job documentation](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145134/Insomnia+-+Submitting+Jobs) says compute-intensive work belongs on compute nodes. The [storage documentation](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145132/Insomnia+-+Storage) says scratch is not backed up; copy important results back to local storage. Never put credentials in scripts or Git.

## 5. Record-coverage experiment branch

Both train-only pilots and both development runs were completed for prompt versions `record-concise-v1` and `record-complete-v1`; see `notes/record-coverage-results.md`. The commands below reproduce those runs or resume incomplete rows. They are not instructions to rerun unchanged outputs. Do not submit a test run with the current complete prompt.

After pushing `main` and `experiment/record-coverage` from the local machine, run the following from the Insomnia login node. Do not rerun the dataset split or regenerate the untouched test fold.

```bash
cd /insomnia001/depts/edu/users/vsp2117/soccer-perspective-summaries
git fetch origin
git switch experiment/record-coverage
git pull --ff-only
source .venv/bin/activate
python3 -m unittest discover -s tests -q
test -s data/processed/matches.jsonl
mkdir -p results
```

Submit the two independent train-only pilots. Each makes six outputs (two matches × neutral/A/B). The first selected train match has 15 recorded events, so it exercises a dense input.

```bash
SPLIT=train CONDITIONS=neutral,fan_A,fan_B CONFIG=configs/record_coverage.json COVERAGE_MODE=concise SMOKE_TEST=2 OUTPUT=results/train_record_concise_smoke.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --time=01:00:00 --mem=32G --cpus-per-task=2 --export=ALL --job-name=record-concise-smoke jobs/baselines.sbatch
SPLIT=train CONDITIONS=neutral,fan_A,fan_B CONFIG=configs/record_coverage.json COVERAGE_MODE=complete SMOKE_TEST=2 OUTPUT=results/train_record_complete_smoke.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --time=01:00:00 --mem=32G --cpus-per-task=2 --export=ALL --job-name=record-complete-smoke jobs/baselines.sbatch
squeue -u "$USER"
```

When both jobs finish, check that each file has six rows, every row is complete, and whether any output reached the 250-token cap:

```bash
python3 - <<'PY'
import json
from pathlib import Path
for mode in ('concise', 'complete'):
    path = Path(f'results/train_record_{mode}_smoke.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    print(mode, 'rows', len(rows), 'complete', sum(r['status'] == 'complete' for r in rows),
          'at token cap', sum(r.get('hit_max_new_tokens') is True for r in rows),
          'word counts', [r.get('output_word_count') for r in rows])
PY
```

Score the pilot against just its two train IDs:

```bash
python3 - <<'PY'
import json
from pathlib import Path
ids = json.loads(Path('data/splits/train.json').read_text())[:2]
Path('results/train_record_smoke_ids.json').write_text(json.dumps(ids))
PY
python3 -m src.sportssum --generations results/train_record_concise_smoke.jsonl --split-ids results/train_record_smoke_ids.json --output results/train_record_concise_smoke_sportssum.csv --details results/train_record_concise_smoke_sportssum.jsonl
python3 -m src.sportssum --generations results/train_record_complete_smoke.jsonl --split-ids results/train_record_smoke_ids.json --output results/train_record_complete_smoke_sportssum.csv --details results/train_record_complete_smoke_sportssum.jsonl
```

If the complete mode reaches the token cap or misses many named events, revise and version the prompt or token limit before a full development run. Do not compare a pilot file against the entire train split. Once the pilot is sound, submit both modes on `dev` using distinct output paths and with `SMOKE_TEST` unset:

```bash
SPLIT=dev CONDITIONS=neutral,fan_A,fan_B CONFIG=configs/record_coverage.json COVERAGE_MODE=concise OUTPUT=results/dev_record_concise.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL --job-name=record-concise-dev jobs/baselines.sbatch
SPLIT=dev CONDITIONS=neutral,fan_A,fan_B CONFIG=configs/record_coverage.json COVERAGE_MODE=complete OUTPUT=results/dev_record_complete.jsonl sbatch -A edu -p short --gres=gpu:l40:1 --export=ALL --job-name=record-complete-dev jobs/baselines.sbatch
```

After both development jobs complete, these CPU-only commands print precision, recall, and F1 for each mode and save separate CSV/JSONL evaluations:

```bash
python3 -m src.sportssum --generations results/dev_record_concise.jsonl --split-ids data/splits/dev.json --output results/dev_record_concise_sportssum.csv --details results/dev_record_concise_sportssum.jsonl
python3 -m src.sportssum --generations results/dev_record_complete.jsonl --split-ids data/splits/dev.json --output results/dev_record_complete_sportssum.csv --details results/dev_record_complete_sportssum.jsonl
```

Freeze settings on dev before generating any `test` summaries. SportsSum metrics cover player names and events, not every field in the record.
