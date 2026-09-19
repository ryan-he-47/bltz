#!/bin/bash
# bltz v2 Stage 1 - enhanced-BPE cache build (CPU job, docs/32 S1).
# Two-stage protocol: SMOKE first (300 docs/file into _v2test dir), inspect,
# then FULL build. Reuses the v1 HF cache parquets when present.
#   sbatch slurm/bltz_cache_v2.sbatch        # full build
#   SMOKE=1 sbatch slurm/bltz_cache_v2.sbatch
#SBATCH --job-name=bltz_cache_v2
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cache_v2_%j.out

set -euo pipefail

PROJ_HOME=/gpfs1/home/yihe47/bltz
SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=$PROJ_HOME/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B   # tokenizer.json only is used

export HF_HOME=$SCRATCH/hf_cache
export TOKENIZERS_PARALLELISM=false   # Rust threads x mp.Pool = oversubscription

mkdir -p "$PROJ_HOME/logs" "$SCRATCH/data" "$HF_HOME"

if [ ! -d "$REPO/bltz" ]; then
  echo "[bootstrap] fetching code ..."
  git clone --depth 50 https://github.com/ryan-he-47/bltz.git "$REPO"
else
  cd "$REPO" && git pull --ff-only || echo "[bootstrap] git pull failed, using existing checkout"
fi
cd "$REPO"
echo "[bootstrap] code at $(git rev-parse --short HEAD 2>/dev/null || echo tarball)"

if [ "${SMOKE:-0}" = "1" ]; then
  $PY -u scripts/build_cache_v2.py configs/cache_v2.yaml \
    --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_smoke" \
    --set seg.tok_dir="$TOK" \
    --set build.max_docs_per_file=300 --set build.workers=4
  echo "[done] SMOKE build at $SCRATCH/data/fineweb10b_v2_smoke — inspect before full build"
else
  $PY -u scripts/build_cache_v2.py configs/cache_v2.yaml \
    --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
    --set seg.tok_dir="$TOK" \
    --set build.workers=4
  touch "$SCRATCH/CACHE_V2_READY"
  echo "[done] v2 cache ready under $SCRATCH/data/fineweb10b_v2"
fi
