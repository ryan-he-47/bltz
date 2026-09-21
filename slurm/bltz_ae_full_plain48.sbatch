#!/bin/bash
# v2: full-data PLAIN AE at 48 dims (deconfounder arm, 2026-09-22 用户预案:
# typo-POC 没炸 -> 补 48维+plain 以隔离 typo 与 d_emb 两个变量)。
# 与 ae_48_256_typo_full 只差 typo.p=0。
#   sbatch slurm/bltz_ae_full_plain48.sbatch
#SBATCH --job-name=bltz_ae_plain48
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ae_plain48_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

cd "$REPO"
git pull --ff-only || echo "[ae_plain48] git pull failed, using existing checkout"

$PY -u scripts/train_ae.py configs/ae.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" --set data.stream=true \
  --set model.d_emb=48 --set typo.p=0.0 \
  --set train.bf16=false --set train.steps=100000 \
  --set train.ckpt_dir="$SCRATCH/ckpts/ae_48_256_plain_full"

echo "[done] plain-48 AE full training; ckpt at $SCRATCH/ckpts/ae_48_256_plain_full"
