#!/bin/bash
# scale 0.5B SEGMENT 1/2 anneal leg (2026-09-27): 2500-step WSD decay tail
# attached to the half-way endpoint. LR 2e-5 (0.1x main peak <= main end LR,
# house rule), decay_frac 0.07 -> the whole leg IS the decay (2e-5 -> 1e-6).
# Produces the annealed MID-TERM model for the "性能是否达到预期" decision
# before burning segment 2. Separate ckpt dir so the flat branch state in
# scale_05b/ckpt_full.pt stays intact for segment 2 continuation.
#   sbatch slurm/bltz_scale_05b_anneal.sbatch   (挂完 half 后手动/依赖链提)
#SBATCH --job-name=bltz_05b_anneal
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/05b_anneal_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/scale_05b_anneal
SRC=$SCRATCH/ckpts/scale_05b/ckpt_full.pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"

if [ ! -f "$SRC" ]; then
  echo "[anneal] FATAL: segment-1 ckpt $SRC missing" >&2
  exit 1
fi
mkdir -p "$CKPT"

RESUME=(--set "train.resume=$SRC")
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[anneal] resuming own state from $CKPT/ckpt_full.pt"
fi

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29522 \
  scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set data.n_patches=4096 \
  --set train.bf16=false --set train.batch=24 \
  --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
  --set model.d_emb=48 --set model.mdn_depth=0 --set model.grad_ckpt=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_typo_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  --set train.lr=2.0e-5 --set train.warmup=1 --set train.steps=35742 \
  --set train.decay_frac=0.07 --set train.final_frac=0.05 \
  --set train.ckpt_every=250 --set train.ckpt_keep=3 --set train.milestone_every=0 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] 0.5B half-anneal finished; mid-term model at $CKPT/best.pt"
