#!/bin/bash
# Deterministic replay of the learnable-arm collapse (2026-09-30 病因分析):
# resume from the HEALTHY milestone ckpt_s0002000 with IDENTICAL config and
# re-aligned rng, run to step 3200 (past the original cascade zone 2260-2938).
# If it re-collapses at the same steps -> deterministic data trigger (harden
# fp16); if it passes -> stochastic state luck (resilient-scaler restart).
#   sbatch slurm/bltz_word_replay.sbatch
#SBATCH --job-name=bltz_word_replay
#SBATCH --partition=stingy
#SBATCH --nodelist=gpu-v100s-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-03:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_replay_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[replay] git pull failed, using existing checkout"

$PY -u scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set model.input_mode=learnable \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.preload=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.batch=36 \
  --set train.steps=3200 \
  --set train.ckpt_keep=2 \
  --set train.resume="$SCRATCH/ckpts/v2_word_1b_learnin/ckpt_s0002000.pt" \
  --set train.ckpt_dir="$SCRATCH/ckpts/v2_word_1b_replay"

echo "[replay] done"
