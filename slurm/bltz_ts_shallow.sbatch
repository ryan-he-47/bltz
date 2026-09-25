#!/bin/bash
# TinyStories MDN-head ablation, ARM shallow (2026-09-24 用户拍板): 单层缓冲(1024)
# current typo-2B recipe (mdn_hidden 1024, plain SiLU MLP), TinyStories cache,
# frozen typo-AE (same as typo-2B). 40k steps ~ 2 epochs of TinyStories.
#   sbatch slurm/bltz_ts_shallow.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_ts_shallow
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ts_shallow_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/ts_shallow

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
  --set data.cache_dir="$SCRATCH/data/tinystories_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=48 --set model.grad_ckpt=true --set model.mdn_depth=1 \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_typo_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  --set train.lr=4.0e-4 --set train.warmup=1000 --set train.steps=40000 \
  --set train.milestone_every=5000 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] ts_shallow finished; log at $CKPT/train.log"
