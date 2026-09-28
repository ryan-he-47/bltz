#!/bin/bash
# pure-BPE cache build (2026-09-28 用户洞察消融): same FineWeb-Edu source,
# segmentation = Qwen3.5 BPE ONLY (deterministic; kills the enhanced-mode
# whole-word-vs-subword ambiguity). Then dump eval pkl. Old mixed cache was
# deleted for quota (local md5-verified backup on E:).
#   sbatch slurm/bltz_cache_v2_bpe.sbatch
#SBATCH --job-name=bltz_cache_bpe
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cache_bpe_%j.out

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
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_bpe" \
  --set seg.tok_dir="$TOK" \
  --set seg.pure_bpe=true \
  --set build.workers=8

$PY -u scripts/dump_eval_pkl.py "$SCRATCH/data/fineweb10b_v2_bpe" \
  "$SCRATCH/data/fineweb10b_v2_bpe/eval_sample.pkl"

echo "[done] pure-BPE cache at $SCRATCH/data/fineweb10b_v2_bpe"
