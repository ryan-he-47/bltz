#!/bin/bash
# v2: full-data AE retrain, 48-256 config, stream mode (docs/32: the S3 frozen
# arm needs an AE trained on the FULL v2 cache, not the 35.8k dryrun pool).
# Requires $SCRATCH/CACHE_V2_READY. 100k steps x batch 256 = 25.6M strings.
# Tiny model (5M) -> fp32 is cheap and sidesteps fp16 scaler noise.
#   sbatch slurm/bltz_ae_full.sbatch
#SBATCH --job-name=bltz_ae_full
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ae_full_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

cd "$REPO"
git pull --ff-only || echo "[ae_full] git pull failed, using existing checkout"

$PY -u scripts/train_ae.py configs/ae.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" --set data.stream=true \
  --set train.bf16=false --set train.steps=100000 \
  --set train.ckpt_dir="$SCRATCH/ckpts/ae_48_256_full"

echo "[done] AE full training; ckpt at $SCRATCH/ckpts/ae_48_256_full"
