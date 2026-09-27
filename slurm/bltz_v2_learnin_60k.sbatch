#!/bin/bash
# learnable-input A/B judge (2026-09-27 用户拍板): lin@60k recipe EXACTLY
# (v2_lin_60k stage-1 trajectory: lr 4e-4, WSD 0.4, 60k, batch 16, S=512,
# 768x12, mdn_depth=0), single variable = model.input_mode=learnable
# (ByteLayerEncoder joint-trained; frozen AE = target generator only).
# Judge vs v2_lin_60k: NLL -61.39 / EM 0.358-0.368 / PPL 1019 / bpb 2.857.
#   sbatch slurm/bltz_v2_learnin_60k.sbatch
#SBATCH --job-name=bltz_v2_learnin
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_learnin_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/v2_learnin_60k

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=48 --set model.mdn_depth=0 --set model.input_mode=learnable \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_typo_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  --set train.steps=60000 \
  --set train.ckpt_every=250 --set train.ckpt_keep=3 --set train.milestone_every=0 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] learnable-input 60k judge run finished; log at $CKPT/train.log"
