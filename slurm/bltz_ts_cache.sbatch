#!/bin/bash
# TinyStories v2 cache build (CPU job, 2026-09-24 对照实验基建):
# download roneneldan/TinyStories train parquets -> rename to the
# build_cache_v2 parquet_dir convention -> enhanced-BPE cache -> eval pkl.
#   sbatch slurm/bltz_ts_cache.sbatch
#SBATCH --job-name=bltz_ts_cache
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ts_cache_%j.out

set -euo pipefail

PROJ_HOME=/gpfs1/home/yihe47/bltz
SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=$PROJ_HOME/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
TS_PQ=$SCRATCH/data/tinystories_parquet
TS_CACHE=$SCRATCH/data/tinystories_v2

export HF_HOME=$SCRATCH/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
git pull --ff-only || echo "[bootstrap] git pull failed, using existing checkout"

echo "[phase 0] downloading TinyStories train parquets ..."
TS_PQ="$TS_PQ" $PY - <<'EOF'
import glob, os, shutil
from huggingface_hub import snapshot_download
d = snapshot_download(repo_id="roneneldan/TinyStories", repo_type="dataset",
                      allow_patterns=["data/train-*.parquet"])
files = sorted(glob.glob(os.path.join(d, "data", "train-*.parquet")))
assert files, "no train parquets found"
os.makedirs(os.environ["TS_PQ"], exist_ok=True)
for i, f in enumerate(files):
    dst = os.path.join(os.environ["TS_PQ"], f"{i:03d}_00000.parquet")
    shutil.copy(f, dst)
    print(f"  {os.path.basename(f)} -> {dst}", flush=True)
print(f"[phase 0] {len(files)} parquets staged", flush=True)
EOF

echo "[phase 1] building cache ..."
$PY -u scripts/build_cache_v2.py configs/cache_v2.yaml \
  --set data.cache_dir="$TS_CACHE" \
  --set seg.tok_dir="$TOK" \
  --set build.parquet_dir="$TS_PQ" \
  --set build.only_files=0,1,2,3 \
  --set build.workers=8

echo "[phase 2] dumping eval pkl ..."
$PY -u scripts/dump_eval_pkl.py "$TS_CACHE" "$TS_CACHE/eval_sample.pkl"

echo "[done] TinyStories cache at $TS_CACHE"
