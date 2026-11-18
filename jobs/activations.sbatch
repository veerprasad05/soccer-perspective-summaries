#!/bin/bash
#SBATCH --job-name=soccer-activations
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/slurm-%j.out
set -euo pipefail
: "${PAIRS:?Set PAIRS to complete record-concise-v1 train-pair JSONL}"
: "${VECTORS:?Set VECTORS to a new .safetensors path}"
: "${METADATA:?Set METADATA to a new .json path}"
CONFIG="${CONFIG:-configs/record_coverage.json}"
LAYERS="${LAYERS:-12,20,26}"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf-cache"
mkdir -p "$HF_HOME" results
source .venv/bin/activate
args=(--config "$CONFIG" --pairs "$PAIRS" --layers "$LAYERS"
      --output "$VECTORS" --metadata "$METADATA")
if [[ -n "${SMOKE_TEST:-}" ]]; then args+=(--smoke-test "$SMOKE_TEST"); fi
python3 -m src.activations "${args[@]}"
