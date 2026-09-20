#!/bin/bash
# v2: full-data TYPO AE retrain (48-256, typo.p=0.1 = 四臂对照的甜点剂量,
# docs/32 工具箱 #2 翻案后立项), stream mode on the v2 cache. End-to-end 校准
# 实验的上游:接下来 bltz_v2_typo_60k.sbatch 用它训骨干,测"骨干预测误差
# 以 typo 形式表达"的终极收益。
#   sbatch slurm/bltz_ae_full_typo.sbatch
#SBATCH --job-name=bltz_ae_typo
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ae_typo_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

cd "$REPO"
git pull --ff-only || echo "[ae_typo] git pull failed, using existing checkout"

$PY -u scripts/train_ae.py configs/ae.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" --set data.stream=true \
  --set model.d_emb=48 --set typo.p=0.1 \
  --set train.bf16=false --set train.steps=100000 \
  --set train.ckpt_dir="$SCRATCH/ckpts/ae_48_256_typo_full"

echo "[done] typo AE full training; ckpt at $SCRATCH/ckpts/ae_48_256_typo_full"
