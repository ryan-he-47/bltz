#!/bin/bash
# AE-verify throughput probe for the SFT cache build (docs/48 §4.3 预算校准).
#   sbatch slurm/bltz_sft_verify_speed.sbatch
#SBATCH --job-name=bltz_sft_vspeed
#SBATCH --partition=stingy
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/sft_vspeed_%j.out

set -euo pipefail
SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

echo "[vspeed] job $SLURM_JOB_ID start $(date '+%F %T')"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -u scripts/probe_sft_verify_speed.py \
  --data-dir "$SCRATCH/data/sft/smol-smoltalk/data" \
  --tok /gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B \
  --ae "$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --n 60
echo "[vspeed] DONE $(date '+%F %T')"
