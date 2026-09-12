#!/bin/bash
# bltz Stage 1 - bltz arm training (V100, fp16+GradScaler).
# batch 16 (bench 544090: batch 32 CUDA-OOM at S=512 on 32G V100; batch 16 =
# 2602ms/step & 23.2GB VRAM, faster than batch 8's 2985ms/step -> 20k ≈ 14.5h).
# LR kept at 4e-4 (batch-8 calibration; under-scaled for 16, safe for iteration).
# Intermediate checkpoints on scratch; best/last + train.log archived to home.
# Time limit / scancel -> SIGTERM 60s early -> trainer saves ckpt_full and exits
# (graceful interrupt, commit 7139054); resubmit to auto-resume.
#   sbatch --dependency=afterok:<cache_jobid> slurm/bltz_train_twin.sbatch
#SBATCH --job-name=bltz_twin
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/twin_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/twin
OUT=/gpfs1/home/yihe47/bltz/ckpt_final

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT" "$OUT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set train.bf16=false --set train.batch=16 \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

cp -f "$CKPT/best.pt" "$CKPT/last.pt" "$CKPT/train.log" "$OUT/" 2>/dev/null || true
echo "[done] bltz arm finished; final artifacts in $OUT"
