#!/bin/bash
# v2 1B long run (2026-09-20 用户拍板): branch from the 60k POC's milestone
# ckpt_s0050000 (weight-only; LR at 50k in the old schedule was ~1.9e-4, so a
# 2e-4 restart peak "对得上", house-rule compliant). **2026-09-21 用户纠正:
# "1B" = 分叉点之后再训 1B patch(+122070 步),不是总曝光凑 1B** —— steps
# 50000+122070=172070;**LR 直接 2e-4 平直(用户拍板允许 ~0.3e-4 小跳变)**,
# decay_frac 0.4 标准(衰减自 103242,末值 2e-5)。
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
  --set train.lr=2.0e-4 --set train.warmup=1 --set train.steps=172070 \
  --set train.decay_frac=0.4 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] v2 1B run finished; log at $CKPT/train.log"
