#!/bin/bash
# FineWeb input-adapter control, ARM linlin (2026-09-25 用户拍板):
# linear head (mdn_depth=0,已赢双档) + LINEAR input adapter (LN+Linear
# 48->768, 去掉 48->2048->GELU->768 的非线性升维). 60k 同轨迹
# (lr 4e-4, WSD 0.4, milestone_every=5000), 对照 v2_lin_60k.
#   sbatch slurm/bltz_v2_linlin_60k.sbatch
#SBATCH --job-name=bltz_v2_linlin
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_linlin_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/v2_linlin_60k
AE=$SCRATCH/ckpts/ae_48_256_typo_full/best.pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=48 --set model.mdn_depth=0 --set model.adapter_linear=true \
  --set train.ae_ckpt="$AE" \
  --set train.ckpt_dir="$CKPT" \
  --set train.steps=60000 --set train.milestone_every=5000 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] linlin 60k finished; log at $CKPT/train.log"
