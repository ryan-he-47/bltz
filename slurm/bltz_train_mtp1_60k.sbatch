#!/bin/bash
# bltz MTP-depth control run (2026-09-16 晚 用户拍板, docs/30): fresh 60k with the
# ONLY supervised-horizon change n_patches_ahead 3 -> 1 (k_max 48 -> 16). External
# results suggest subword-level MTP can hurt learning in small models (ours is
# ~124M); this arm tests whether that holds here. LR schedule identical to the
# 60k run (WSD: lr 4e-4, warmup 1000, decay_frac 0.4, final_frac 0.1 — all from
# configs/default.yaml). Known confound vs the 60k arm: head is now film (user
# decision 2026-09-16: film is the permanent default), 60k was concat — POC
# qualitative reading applies (docs/30 §判读).
#   sbatch slurm/bltz_train_mtp1_60k.sbatch   (重提交自动续自己的 ckpt)
#SBATCH --job-name=bltz_60k_mtp1
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/train60k_mtp1_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/twin60k_mtp1
OUT=/gpfs1/home/yihe47/bltz/ckpt_final/60k_mtp1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT" "$OUT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own 60k-mtp1 state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set train.bf16=false --set train.batch=16 --set train.steps=60000 \
  --set loss.n_patches_ahead=1 --set model.k_max=16 \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

cp -f "$CKPT/best.pt" "$CKPT/last.pt" "$CKPT/train.log" "$OUT/" 2>/dev/null || true
echo "[done] 60k-mtp1 run finished; final artifacts in $OUT"
