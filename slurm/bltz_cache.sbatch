#!/bin/bash
# bltz Stage 1 - cache build (CPU job): clone/pull code -> build byte patch
# cache -> build token cache. Heavy data stays on scratch; home holds code/logs.
# Submit: sbatch bltz_cache.sbatch   (dependency-free, first job of the stage)
#SBATCH --job-name=bltz_cache
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cache_%j.out

set -euo pipefail

PROJ_HOME=/gpfs1/home/yihe47/bltz
SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=$PROJ_HOME/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

export HF_HOME=$SCRATCH/hf_cache          # HF downloads must NOT land in home quota
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PROJ_HOME/logs" "$SCRATCH/data" "$HF_HOME"

# --- code bootstrap: git (sync channel) first, scp'd tarball as fallback ------
if [ ! -d "$REPO/bltz" ]; then
  echo "[bootstrap] fetching code ..."
  if ! git clone --depth 50 https://github.com/ryan-he-47/bltz.git "$REPO"; then
    echo "[bootstrap] git clone failed, falling back to tarball"
    mkdir -p "$REPO"
    tar -xzf /gpfs1/scratch/yihe47/bltz_bootstrap.tar.gz -C "$REPO"
  fi
else
  cd "$REPO" && git pull --ff-only || echo "[bootstrap] git pull failed, using existing checkout"
fi
cd "$REPO"
echo "[bootstrap] code at $(git rev-parse --short HEAD 2>/dev/null || echo tarball)"

# --- byte patch cache (14 parquet, ~30GB download, parallel segmentation) -----
# workers=4 (2026-09-12 lesson: 8 workers under the 48G cgroup + GPFS page cache
# pressure -> tail workers OOM-killed -> mp.Pool hangs forever. 4 workers keeps
# anon+cache far from the limit; remaining shards rebuild in ~15 min.)
$PY -u scripts/build_cache.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set build.workers=4

# --- token cache (parquet reused from HF cache) --------------------------------
$PY -u scripts/build_token_cache.py configs/baseline.yaml \
  --set data.cache_dir="$SCRATCH/data/tokengpt2"

touch "$SCRATCH/CACHE_READY"
echo "[done] caches ready under $SCRATCH/data"
