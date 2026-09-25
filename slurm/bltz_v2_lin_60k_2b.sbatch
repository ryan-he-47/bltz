#!/bin/bash
# FineWeb linear-head validation, TWO STAGES in one job (2026-09-24 用户拍板):
# stage 1 = exact typo-60k trajectory with --set model.mdn_depth=0
#           (60k steps, lr 4e-4, WSD 0.4, milestone_every=5000)
# stage 2 = exact typo-2B trajectory branching from stage 1's OWN 50k milestone
#           (+2B units: steps=294140, lr 2e-4, warmup=1)
#   sbatch slurm/bltz_v2_lin_60k_2b.sbatch   (重提交自动续命:各阶段续自己的 ckpt)
#SBATCH --job-name=bltz_v2_lin
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_lin_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT1=$SCRATCH/ckpts/v2_lin_60k
CKPT2=$SCRATCH/ckpts/v2_lin_2b
AE=$SCRATCH/ckpts/ae_48_256_typo_full/best.pt
MILE=$CKPT1/ckpt_s0050000.pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT1" "$CKPT2"

# ---------- stage 1: linear-head 60k POC (typo-60k trajectory) ----------
if [ ! -f "$MILE" ]; then
  RESUME=()
  if [ -f "$CKPT1/ckpt_full.pt" ]; then
    RESUME=(--set "train.resume=$CKPT1/ckpt_full.pt")
    echo "[stage1] resuming own state from $CKPT1/ckpt_full.pt"
  fi
  $PY -u scripts/train_bltz_v2.py configs/v2.yaml \
    --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
    --set train.bf16=false --set train.batch=16 \
    --set model.d_emb=48 --set model.mdn_depth=0 \
    --set train.ae_ckpt="$AE" \
    --set train.ckpt_dir="$CKPT1" \
    --set train.steps=60000 --set train.milestone_every=5000 \
    ${RESUME[@]+"${RESUME[@]}"}
else
  echo "[stage1] milestone $MILE already exists, skip to stage 2"
fi

# ---------- stage 2: +2B units from the 50k milestone (typo-2B trajectory) ----------
if [ ! -f "$MILE" ]; then
  echo "[stage2] FATAL: stage-1 milestone $MILE missing, aborting" >&2
  exit 1
fi
RESUME=(--set "train.resume=$MILE")
if [ -f "$CKPT2/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT2/ckpt_full.pt")
  echo "[stage2] resuming own state from $CKPT2/ckpt_full.pt"
fi
$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=48 --set model.mdn_depth=0 \
  --set train.ae_ckpt="$AE" \
  --set train.ckpt_dir="$CKPT2" \
  --set train.lr=2.0e-4 --set train.warmup=1 --set train.steps=294140 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] linear-head 60k + 2B two-stage run finished"
