#!/bin/bash
#SBATCH --job-name=soccer-activations
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/slurm-%j.out
# Placeholder only: activation extraction is not implemented. Do not submit.
set -euo pipefail
printf '%s\n' 'Activation extraction pending source-grounded train pairs and implementation' >&2
exit 2
