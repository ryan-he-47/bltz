#!/bin/bash
# v2 TYPO arm long run to compute-optimal spec (2026-09-22 用户拍板: 我们第
# 一个按计算最优训练的模型): branch from typo POC milestone ckpt_s0050000
# (LR at 50k ~1.9e-4, restart peak 2e-4 "对得上"), +2B units = +244140 steps
# (batch16*512), total steps 294140, WSD decay_frac 0.4, warmup=1.
# Frozen 48-256 typo-AE (p=0.1). ~30h at 0.45s/step.
#   sbatch slurm/bltz_v2_typo_2b.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_v2_typo2b
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_typo2b_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/v2_typo_2b
MILE=$SCRATCH/ckpts/v2_typo_poc_60k/ckpt_s0050000.pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=(--set "train.resume=$MILE")
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own typo-2b state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=48 \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_typo_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  --set train.lr=2.0e-4 --set train.warmup=1 --set train.steps=294140 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] v2 typo 2B run finished; log at $CKPT/train.log"
