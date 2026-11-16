#!/bin/bash
#SBATCH --job-name=soccer-steering
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/slurm-%j.out
# Placeholder only: steering is not implemented. Do not submit.
set -euo pipefail
printf '%s\n' 'Steering pending learned vector implementation and validation' >&2
exit 2
