#!/bin/bash
#SBATCH --job-name=soccer-smoke
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/slurm-%j.out
# Submit with verified: sbatch -A YOUR_ACCOUNT -p YOUR_GPU_PARTITION --gres=YOUR_VERIFIED_GPU_REQUEST jobs/smoke.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SLURM_SUBMIT_DIR/.hf-cache"
mkdir -p "$HF_HOME"
source .venv/bin/activate
python3 -m src.models.gpu_smoke --load-model
