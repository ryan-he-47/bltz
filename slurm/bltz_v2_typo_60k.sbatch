#!/bin/bash
# v2 backbone POC, TYPO-AE arm (docs/32: 端到端校准验证): frozen 48-256
# typo-AE (p=0.1) + 113M backbone + MDN head, 60k steps WSD — same as the
# plain-AE POC (574099/575524 line) otherwise. REQUIRES: CACHE_V2_READY +
# ae_48_256_typo_full/best.pt. NOTE(对照口径,2026-09-21 用户记): 本臂同时改
# typo+48维两个变量——若训炸,补 32维+typo 版本;没炸,补 48维+plain 版本。
#   sbatch slurm/bltz_v2_typo_60k.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_v2_typo60k
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_typo60k_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/v2_typo_poc_60k

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own typo-poc state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=48 \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_typo_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] v2 typo 60k POC finished; log at $CKPT/train.log"
