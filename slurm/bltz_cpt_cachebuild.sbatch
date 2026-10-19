#!/bin/bash
# CPT word-cache build (docs/43 §2, 2026-10-18): segment_word re-tokenize the
# new corpus (DCLM / FineWeb-Edu-dedup / anneal sources) into shard caches,
# then assemble the mixed dirs via symlinks.
# 10-18 用户纪律: 长任务一律 sbatch — this job IS the cache build; nothing
# heavy runs on the login node. Raw text parquets are deleted per-source right
# after their cache finishes (re-downloadable; 300G quota, docs/43 §1.1).
#   sbatch --dependency=afterok:<dl_job> slurm/bltz_cpt_cachebuild.sbatch
#SBATCH --job-name=bltz_cpt_cache
#SBATCH --partition=tiny
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cpt_cache_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
DATA=$SCRATCH/data

cd "$REPO"
git pull --ff-only || echo "[cache] git pull failed, using existing checkout"

build() {  # src_dir offset cache_dir [extra --set args...]
  echo "===== build $1 -> $3 (offset $2) ====="
  local SRC=$1 OFF=$2 DST=$3; shift 3
  $PY -u scripts/build_cache_v2.py configs/cache_v2.yaml \
    --set data.cache_dir="$DST" \
    --set build.src_dir="$SRC" --set build.shard_offset="$OFF" \
    --set build.workers=4 --set seg.mode=word --set seg.tok_dir="$TOK" "$@"
}

# ---- main-mix pools (60 DCLM : 40 Edu by construction) ----
build "$DATA/cpt_dclm"      0 "$DATA/cpt_cache_dclm"
rm -rf "$DATA/cpt_dclm"
build "$DATA/cpt_fweb_edu"  0 "$DATA/cpt_cache_edu"
rm -rf "$DATA/cpt_fweb_edu"

# ---- anneal pool (offsets must match scripts/assemble_cpt_caches.py) ----
POOL=$DATA/cpt_cache_anneal_pool
build "$DATA/cpt_anneal/dolmino_dclm"          0   "$POOL"
build "$DATA/cpt_anneal/dolmino_pes2o"         50  "$POOL"
build "$DATA/cpt_anneal/dolmino_wiki"          100 "$POOL"
build "$DATA/cpt_anneal/dolmino_stackexchange" 150 "$POOL"
build "$DATA/cpt_anneal/stack_edu_python"      200 "$POOL"
build "$DATA/cpt_anneal/stack_edu_javascript"  210 "$POOL"
# finemath 切小 shard(max_docs 6 万/片 ≈ 数千万 units/片): 整 shard 组装
# 粒度太粗, 数学会冲破 5% 拍板线(2026-10-19)。(markdown 已从配比去掉,
# 见 docs/43 §1.1 二轮换源注记; 勿再加 build 行。)
build "$DATA/cpt_anneal/finemath"              230 "$POOL" \
  --set build.max_docs_per_file=60000
rm -rf "$DATA/cpt_anneal"

# ---- assemble mixed dirs + unit report + peek ----
$PY -u scripts/assemble_cpt_caches.py --data "$DATA"

echo "[done] CPT caches built; log above"
