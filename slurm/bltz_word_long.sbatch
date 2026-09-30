#!/bin/bash
# Word-arm CONTINUATION to Chinchilla-optimal (2026-09-30 用户拍板):
# resume from pre-decay milestone ckpt_s0054000 (54000 < 54266 decay_start of
# the parent 1.25B run), new absolute-step WSD: steps=112000 (120M*20 tokens
# ≈ 10.3B bytes ≈ 2.06B units), decay_frac 0.375 (decay 70k->112k, 42k steps
# ≈ 0.77B units tail — 3x the 1B run's tail), final_frac=0 (full anneal),
# rewarmup 500 (fresh-Adam guard; peak 4e-4 <= parent end LR, house rule OK).
# First run with block-level shard rotation (用户拍板: 服务器积德) — mem 48G.
# Checkpoint policy (用户拍板 09-30 修订): rolling 3 full + permanent
# milestones every 10k steps (branch points for long runs) + best.pt.
#   sbatch slurm/bltz_word_long.sbatch
#SBATCH --job-name=bltz_word_long
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_long_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[word_long] git pull failed, using existing checkout"

$PY -u scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.rotate_shards.resident=4 \
  --set data.rotate_shards.every=2000 \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.fp16_init_scale=1024.0 \
  --set train.fp16_growth_interval=500 \
  --set train.rewarmup=500 \
  --set train.batch=36 \
  --set train.steps=112000 \
  --set train.decay_frac=0.375 \
  --set train.final_frac=0.0 \
  --set train.ckpt_keep=3 \
  --set train.milestone_every=10000 \
  --set train.resume="$SCRATCH/ckpts/v2_word_1b/ckpt_s0054000.pt" \
  --set train.ckpt_dir="$SCRATCH/ckpts/v2_word_long"

echo "[word_long] done"
