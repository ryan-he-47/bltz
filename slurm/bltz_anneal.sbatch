#!/bin/bash
# bltz anneal probe (2026-09-16 用户拍板,河谷探底): pause the 2B run, branch
# its latest ckpt into a 10k-step linear anneal from the CURRENT LR down to
# ~2e-5 (peak = current LR -> no jump, house-rule compliant). Answers: is the
# plateau LR-bound (loss dives) or basin-bound (barely moves)?
#   sbatch slurm/bltz_anneal.sbatch
#SBATCH --job-name=bltz_anneal
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/anneal_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/anneal_140k
MAIN=$SCRATCH/ckpts/boundary_2b

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[anneal] git pull failed, using existing checkout"
mkdir -p "$CKPT"

if [ ! -f "$CKPT/frozen.pt" ]; then
  cp "$MAIN/ckpt_full.pt" "$CKPT/frozen.pt"  # main run is paused (scancel); rotation is atomic
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
STEPS=$((STEP + 10000))
DF=$($PY -c "print(10000.0 / $STEPS)")
echo "[anneal] resume@$STEP lr=$LR -> 10k linear anneal, steps=$STEPS decay_frac=$DF"

$PY -u scripts/train_bltz.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set train.bf16=false --set train.batch=16 \
  --set train.ckpt_dir="$CKPT" --set train.resume="$CKPT/frozen.pt" \
  --set train.lr="$LR" --set train.warmup=1 --set train.decay_frac="$DF" \
  --set train.steps="$STEPS" --set train.log_every=25

echo "[done] anneal finished; log at $CKPT/train.log"
