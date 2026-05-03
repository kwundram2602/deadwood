#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=requeue-gpu,gpu2080,gpua100,gputitanrtx,gpu3090,gpuhgx
#SBATCH --time=0-08:00:00
#SBATCH --job-name=deadwood_train
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/tmp/%u/deadwood/train_%j.log

# ---------------------------------------------------------------------------
# deadwood crown segmentation — training script for Palma HPC cluster
# Submit from repo root: sbatch deadwood/hpc/train_torch.sh [--config <cfg>]
# ---------------------------------------------------------------------------

module purge
ml palma/2023a foss/2023a
ml uv

# Paths on Palma — adjust $WORK to your actual workspace
REPO="$HOME/InnoLab_DL"
CODE="$REPO/deadwood"
CONFIG="${1:-configs/crown_ms.yaml}"   # override: sbatch train_torch.sh configs/my.yaml

mkdir -p /scratch/tmp/$USER/deadwood

cd "$CODE"

# Sync the uv environment (installs CUDA wheels on Linux automatically via pyproject.toml markers)
uv sync

# Run training
uv run python scripts/train.py \
    --config "$CONFIG" \
    --working_dir .

echo "Job $SLURM_JOB_ID finished at $(date +%Y.%m.%d-%H:%M:%S)"
