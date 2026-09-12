#!/bin/bash
# bltz Stage 1 - token baseline arm (GPT-2 tokenizer, backbone-matched).
# Same discipline as the bltz arm: fp16, batch 32, scratch ckpts, home archive.
#   sbatch --dependency=afterok:<cache_jobid> slurm/bltz_train_baseline.sbatch
#SBATCH --job-name=bltz_baseline
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/baseline_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/baseline_twin
OUT=/gpfs1/home/yihe47/bltz/ckpt_final

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT" "$OUT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_token_baseline.py configs/baseline.yaml \
  --set data.cache_dir="$SCRATCH/data/tokengpt2" \
  --set train.bf16=false --set train.batch=32 \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

cp -f "$CKPT/best.pt" "$CKPT/last.pt" "$CKPT/train.log" "$OUT/" 2>/dev/null || true
echo "[done] baseline arm finished; final artifacts in $OUT"
