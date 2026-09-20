#!/bin/bash
# v2 1B long run (2026-09-20 用户拍板): branch from the 60k POC's milestone
# ckpt_s0050000 (weight-only; LR at 50k in the old schedule was ~1.9e-4, so a
# 2e-4 restart peak "对得上", house-rule compliant). New WSD: 122070 steps total
# (~1.0B patches at batch16*512, i.e. +72k steps ≈ 8h), decay_frac 0.4,
# warmup=1 (resume branch). Goal: keep probing the NLL floor at scale.
#   sbatch slurm/bltz_v2_1b.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_v2_1b
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_1b_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/v2_1b
MILE=$SCRATCH/ckpts/v2_poc_60k/ckpt_s0050000.pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=(--set "train.resume=$MILE")
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own 1b state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=32 \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_32_256_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  --set train.lr=2.0e-4 --set train.warmup=1 --set train.steps=122070 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] v2 1B run finished; log at $CKPT/train.log"
