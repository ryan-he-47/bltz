#!/bin/bash
# v2 backbone POC, PLAIN-48 deconfounder arm (2026-09-22 用户预案): frozen
# 48-256 PLAIN AE + 113M backbone + MDN head, 60k steps WSD. 与 typo-POC
# (576436) 唯一差异 = typo.p(0.0 vs 0.1)——隔离 typo 与 d_emb 两变量。
# REQUIRES: ae_48_256_plain_full/best.pt.
#   sbatch slurm/bltz_v2_plain48_60k.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_v2_plain48
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_plain48_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/v2_plain48_poc_60k

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own plain48-poc state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=48 \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_plain_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] v2 plain48 60k POC finished; log at $CKPT/train.log"
