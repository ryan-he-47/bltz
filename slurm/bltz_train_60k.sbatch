#!/bin/bash
# bltz POC - 60k long run (2026-09-13 用户拍板,gcos 修正测量判良性后).
# Fresh WSD(60000), NOT a resume of the 20k run (LR-jump + 续训 house rule).
# batch16/fp16/LR 4e-4; bench 实测 2602ms/step -> 60k ≈ 43.4h(5 天墙内).
# 里程碑 weight-only 每 2000 步(全程在,供 gcos 轨迹复查 + decay-on-demand)。
#   sbatch slurm/bltz_train_60k.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_60k
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/train60k_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/twin60k
OUT=/gpfs1/home/yihe47/bltz/ckpt_final/60k

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT" "$OUT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own 60k state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set train.bf16=false --set train.batch=16 --set train.steps=60000 \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

cp -f "$CKPT/best.pt" "$CKPT/last.pt" "$CKPT/train.log" "$OUT/" 2>/dev/null || true
echo "[done] 60k run finished; final artifacts in $OUT"
