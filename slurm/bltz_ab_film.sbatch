#!/bin/bash
# bltz POC - head A/B, FiLM arm (2026-09-14 用户拍板): new encoder
# (inducing=8, enc_heads=4, committed in configs/default.yaml) + FiLM head.
# 20k steps, same recipe as the 60k run (batch16/fp16/LR 4e-4/WSD).
# Control arm: slurm/bltz_ab_concat.sbatch (identical except head_type).
#   sbatch slurm/bltz_ab_film.sbatch
#SBATCH --job-name=bltz_ab_film
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ab_film_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/ab_film
OUT=/gpfs1/home/yihe47/bltz/ckpt_final/ab_film

cd "$REPO"
git pull --ff-only || echo "[ab] git pull failed, using existing checkout"
mkdir -p "$CKPT" "$OUT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[ab] resuming own state from $CKPT/ckpt_full.pt"
fi

# --- memory guards (2026-09-14 OOM lesson): fused FiLM head + head grad
# checkpointing (retention bounded by chunk, not total queries — batches pack
# by PATCH count so query count floats with byte density) + expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PY -u scripts/train_bltz.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set train.bf16=false --set train.batch=16 --set train.steps=20000 \
  --set model.head_type=film --set loss.head_grad_ckpt=true \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

cp -f "$CKPT/best.pt" "$CKPT/last.pt" "$CKPT/train.log" "$OUT/" 2>/dev/null || true
echo "[done] ab_film finished; final artifacts in $OUT"
