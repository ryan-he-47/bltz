#!/bin/bash
# v2 word-arm FORMAL run, LEARNABLE arm (2026-09-29 用户严谨对照规格):
# 与 frozen 臂唯一区别 = input_mode=learnable (in_enc 梯度联通,
# AE 同构镜像 + 去瓶颈 proj+LN; 目标仍由冻结 AE 提供). batch 36 双臂对齐
# (V100 smoke: 27.22GiB peak, 4.78G headroom), 同预算 1.25B units
# = 67832 steps @ ~0.76s/step ~ 14.3h.
#   sbatch slurm/bltz_word_1b_learnin.sbatch
#SBATCH --job-name=bltz_word_1b_learnin
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_1b_learnin_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[word_1b_learnin] git pull failed, using existing checkout"

$PY -u scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set model.input_mode=learnable \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.preload=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.batch=36 \
  --set train.steps=67832 \
  --set train.ckpt_keep=4 \
  --set train.ckpt_dir="$SCRATCH/ckpts/v2_word_1b_learnin"

echo "[word_1b_learnin] done"
