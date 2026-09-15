#!/bin/bash
# bltz T1'-era boundary probe (2026-09-15 用户拍板): encoder(inducing8/heads4,
# grouped 长度分组)+ FiLM head(默认), **2B patch 预算**长跑,正式探模型能力
# 边界。2e9 patches / (16x512) = 244000 steps ≈ 12.1B bytes(60k 的 4.3x;
# 数据占缓存 26%,无重复)。batch 16(显存效率优化后优先);~80h,5 天墙内。
#   sbatch slurm/bltz_train_2b.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_2b
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/train2b_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/boundary_2b
OUT=/gpfs1/home/yihe47/bltz/ckpt_final/boundary_2b

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT" "$OUT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own 2B state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set train.bf16=false --set train.batch=16 --set train.steps=244000 \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

cp -f "$CKPT/best.pt" "$CKPT/last.pt" "$CKPT/train.log" "$OUT/" 2>/dev/null || true
echo "[done] 2B boundary run finished; final artifacts in $OUT"
