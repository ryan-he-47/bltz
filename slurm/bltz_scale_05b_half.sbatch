#!/bin/bash
# scale 0.5B SEGMENT 1/2 (2026-09-27 用户拍板): first half of one full pass
# (33242 steps = 3.19M seqs / 2 / 48 global), 2x V100 DDP, per-GPU batch 24
# (smoke: 24.6GiB peak, 12.65 s/step; 6GB-headroom rule), S=4096.
# LR 2e-4 FLAT (decay_frac=0 -> end LR = peak, so segment 2 may restart at
# 2e-4 under the house rule). Rolling ckpt_keep=3 only (disk quota 拍板),
# no permanent milestones. Wall-out is fine: resubmit resumes from ckpt_full.
#   sbatch slurm/bltz_scale_05b_half.sbatch
#SBATCH --job-name=bltz_05b_half
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/05b_half_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/scale_05b

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own state from $CKPT/ckpt_full.pt"
fi

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29521 \
  scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set data.n_patches=4096 \
  --set train.bf16=false --set train.batch=24 \
  --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
  --set model.d_emb=48 --set model.mdn_depth=0 --set model.grad_ckpt=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_typo_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  --set train.lr=2.0e-4 --set train.warmup=2000 --set train.steps=33242 \
  --set train.decay_frac=0.0 \
  --set train.ckpt_every=250 --set train.ckpt_keep=3 --set train.milestone_every=0 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] 0.5B segment-1 (half) finished; log at $CKPT/train.log"
