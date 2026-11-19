#!/bin/bash
#SBATCH --job-name=soccer-steering
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/slurm-%j.out
set -euo pipefail
: "${VECTORS:?Set VECTORS to trained .safetensors path}"
: "${METADATA:?Set METADATA to its metadata JSON path}"
: "${BASELINES:?Set BASELINES to concise neutral baseline JSONL}"
: "${LAYER:?Set LAYER to a vector layer index}"
: "${STRENGTHS:?Set STRENGTHS to comma-separated positive coefficients}"
: "${OUTPUT:?Set OUTPUT to a new steering JSONL path}"
CONFIG="${CONFIG:-configs/record_coverage.json}"
SPLIT="${SPLIT:-dev}"
RANDOM_SEEDS="${RANDOM_SEEDS:-17,29,43}"
NEUTRAL_PROMPT_MODE="${NEUTRAL_PROMPT_MODE:-concise}"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf-cache"
mkdir -p "$HF_HOME" results
source .venv/bin/activate
args=(--config "$CONFIG" --split "$SPLIT" --vectors "$VECTORS"
      --metadata "$METADATA" --baselines "$BASELINES" --layer "$LAYER"
      --strengths "$STRENGTHS" --random-seeds "$RANDOM_SEEDS"
      --neutral-prompt-mode "$NEUTRAL_PROMPT_MODE" --output "$OUTPUT")
if [[ -n "${SMOKE_TEST:-}" ]]; then args+=(--smoke-test "$SMOKE_TEST"); fi
python3 -m src.steering "${args[@]}"
