#!/bin/bash
# v2 backbone POC (docs/32 S3, frozen arm): frozen 48-256 AE + 113M backbone
# + MDN head, 60k steps WSD (2026-09-18 拍板: skip 20k first-read; early-stop
# anytime, milestones keep branch points). REQUIRES: CACHE_V2_READY + the AE
# full ckpt ($SCRATCH/ckpts/ae_48_256_full/best.pt) — submit only after both.
#   sbatch slurm/bltz_v2_60k.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_v2_60k
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_60k_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/v2_poc_60k

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own v2 state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] v2 60k POC finished; log at $CKPT/train.log"
