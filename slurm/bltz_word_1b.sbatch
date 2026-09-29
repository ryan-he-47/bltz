#!/bin/bash
# v2 word-arm FORMAL run, FROZEN control (2026-09-29 拍板): word 臂缓存 +
# ae_48_256_word_full + linear MDN + 在线核验回退. 1B const + 0.25B decay
# (WSD decay_frac 0.2), batch 36 双臂对齐 (用户拍板; smoke 实测斜率
# 0.370GiB/batch -> ~15GiB 峰值, 余量充裕), 1.25B units = 67832 steps
# @ ~0.46s/step ~ 8.7h.
#   sbatch slurm/bltz_word_1b.sbatch
#SBATCH --job-name=bltz_word_1b
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_1b_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[word_1b] git pull failed, using existing checkout"

$PY -u scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.preload=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.batch=36 \
  --set train.steps=67832 \
  --set train.ckpt_keep=4 \
  --set train.ckpt_dir="$SCRATCH/ckpts/v2_word_1b"

echo "[word_1b] done"
