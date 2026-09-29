#!/bin/bash
# Full word-arm cache (2026-09-29): seg.mode=word (merge=prev trailing-space
# + per-digit + BPE fallback), all 14 FineWeb-Edu parquets. OOM 防范: pure-BPE
# build's 8-worker hang was worker OOM -> mp.Pool deadlock; word mode is
# per-doc python (single-core/worker), workers=4 halves the footprint.
#   sbatch slurm/bltz_cache_word_full.sbatch
#SBATCH --job-name=bltz_cache_word
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cache_word_%j.out

set -euo pipefail

PROJ_HOME=/gpfs1/home/yihe47/bltz
SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=$PROJ_HOME/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

export HF_HOME=$SCRATCH/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
git pull --ff-only || echo "[bootstrap] git pull failed, using existing checkout"

$PY -u scripts/build_cache_v2.py configs/cache_v2.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set seg.tok_dir="$TOK" \
  --set seg.mode=word \
  --set build.workers=4

$PY -u scripts/dump_eval_pkl.py "$SCRATCH/data/fineweb10b_v2_word" \
  "$SCRATCH/data/fineweb10b_v2_word/eval_sample.pkl"

echo "[done] word-arm cache at $SCRATCH/data/fineweb10b_v2_word"
