#!/bin/bash
# sigma-floor anneal, ARM sig01 (2026-09-24 用户拍板, sigma 问题优先):
# typo-2B ckpt_s0290000 分叉, weight-only resume, model.sigma_floor 0.05->0.01,
# 15k steps flat-ish anneal 2e-5 -> 2e-6 (steps=305000, decay_frac=0.05).
# 预测(写死): 原生 NLL 改善(含机械项), edit<=2 混淆尾改善, EM 小涨,
# snap-PPL 校准 T 恶化. 对照臂 = bltz_v2_sig05_anneal.sbatch (sigma_floor 不变).
#   sbatch slurm/bltz_v2_sig01_anneal.sbatch
#SBATCH --job-name=bltz_v2_sig01
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/v2_sig01_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/v2_typo2b_sig01
MILE=$SCRATCH/ckpts/v2_typo_2b/ckpt_s0290000.pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=(--set "train.resume=$MILE")
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own sig01 state from $CKPT/ckpt_full.pt"
fi

$PY -u scripts/train_bltz_v2.py configs/v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.bf16=false --set train.batch=16 \
  --set model.d_emb=48 --set model.sigma_floor=0.01 \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_typo_full/best.pt" \
  --set train.ckpt_dir="$CKPT" \
  --set train.lr=2.0e-5 --set train.warmup=1 --set train.steps=305000 \
  --set train.decay_frac=0.05 --set train.final_frac=0.1 \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] v2 sig01 anneal finished; log at $CKPT/train.log"
