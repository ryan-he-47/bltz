#!/bin/bash
# bltz 25k anneal-to-zero probe (2026-09-16 晚 用户拍板, docs/27 §5): same frozen
# ckpt@140750 as the 10k arm (565575), but 25k steps and LR linear all the way to
# EXACTLY 0 (train.final_frac=0.0). Purpose: the 10k arm (2.52) had not flattened
# at 2.2e-5 — maybe under-cooled. Annealing to 0 gets the true basin floor and,
# with the same start point, gives an anneal-length controlled comparison.
#   sbatch slurm/bltz_anneal25k.sbatch
#SBATCH --job-name=bltz_anneal25k
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/anneal25k_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/anneal25k_140k
SRC=$SCRATCH/ckpts/anneal_140k/frozen.pt   # frozen @140750, identical to the 10k arm's start

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[anneal25k] git pull failed, using existing checkout"
mkdir -p "$CKPT"

if [ ! -f "$CKPT/frozen.pt" ]; then
  cp "$SRC" "$CKPT/frozen.pt"
fi

# resume step + the main run's WSD LR at that step (decay_start 39040, 244k total)
read -r STEP LR <<< "$($PY -c "
import torch
s = torch.load('$CKPT/frozen.pt', map_location='cpu', weights_only=False)
step = int(s['step']) + 1
peak, floor, d0, total = 4e-4, 4e-5, 39040, 244000
lr = peak if step < d0 else peak + (floor - peak) * min((step - d0) / (total - d0), 1.0)
print(step, lr)
")"
STEPS=$((STEP + 25000))
DF=$($PY -c "print(25000.0 / $STEPS)")
echo "[anneal25k] resume@$STEP lr=$LR -> 25k linear anneal to 0, steps=$STEPS decay_frac=$DF final_frac=0.0"

$PY -u scripts/train_bltz.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set train.bf16=false --set train.batch=16 \
  --set train.ckpt_dir="$CKPT" --set train.resume="$CKPT/frozen.pt" \
  --set train.lr="$LR" --set train.warmup=1 --set train.decay_frac="$DF" \
  --set train.final_frac=0.0 \
  --set train.steps="$STEPS" --set train.log_every=25

echo "[done] anneal25k finished; log at $CKPT/train.log"
