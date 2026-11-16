#!/bin/bash
#SBATCH --job-name=soccer-baselines
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/slurm-%j.out
# Submit with verified: sbatch -A YOUR_ACCOUNT -p YOUR_GPU_PARTITION --gres=YOUR_VERIFIED_GPU_REQUEST jobs/baselines.sbatch
set -euo pipefail
: "${SPLIT:?Set SPLIT=train,dev,or test}"
: "${CONDITIONS:?Set CONDITIONS=neutral,fan_A,fan_B or fan_A,fan_B}"
: "${OUTPUT:?Set OUTPUT path}"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf-cache"
mkdir -p "$HF_HOME"
source .venv/bin/activate
python3 -m src.generate --config configs/baselines.json --split "$SPLIT" --conditions "$CONDITIONS" --output "$OUTPUT"
