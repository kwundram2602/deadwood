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
CONFIG="${1:-configs/train_config/crown_ms.yaml}"   # override: sbatch train_torch.sh configs/my.yaml
# output_dir is set inside the YAML as an absolute path (e.g. /scratch/tmp/$USER/deadwood/out)

cd "$CODE"

# Sync the uv environment (installs CUDA wheels on Linux automatically via pyproject.toml markers)
uv sync

# Run training — outputs land in <output_dir>/<experiment_id>/ (from config)
uv run python scripts/train.py \
  --config "$CONFIG" \
  --working_dir .
echo "Training finished at $(date +%Y.%m.%d-%H:%M:%S)"
echo ""
echo "To evaluate, run:"
echo "  uv run python scripts/evaluate.py --config \$CONFIG --weights \$OUT_DIR/<experiment_id>/ft_best.pt --working_dir ."

echo "Job $SLURM_JOB_ID finished at $(date +%Y.%m.%d-%H:%M:%S)"
