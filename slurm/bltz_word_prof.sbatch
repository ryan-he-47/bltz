#!/bin/bash
# Phase-profiling diagnostic (2026-09-29 slowdown hunt): 400-step frozen run
# with BLTZ_PROF=1 on the (idle) stingy V100 -> per-phase wall-time split
# (batch/fwd/bwd/gn/opt/io) on a clean server CPU, isolating code-path cost
# from production-node environment. ~10 min.
#   sbatch slurm/bltz_word_prof.sbatch
#SBATCH --job-name=bltz_word_prof
#SBATCH --partition=stingy
#SBATCH --nodelist=gpu-v100s-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_prof_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export BLTZ_PROF=1

cd "$REPO"
git pull --ff-only || echo "[prof] git pull failed, using existing checkout"

$PY -u scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.preload=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.batch=36 \
  --set train.steps=400 \
  --set train.ckpt_every=100000 \
  --set train.milestone_every=0 \
  --set train.ckpt_dir="$SCRATCH/ckpts/v2_word_prof"

echo "[prof] done"
