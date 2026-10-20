#!/bin/bash
# AE reconstruction probe on the new CPT caches vs old-corpus baseline
# (2026-10-20 用户问: AE 在 DCLM 上的编码质量; 生产侧 flag 率已平,
# 这里给直测 EM/CE/bacc)。tiny 分区, 分钟级。
#   sbatch slurm/bltz_probe_ae_shards.sbatch
#SBATCH --job-name=bltz_ae_probe
#SBATCH --partition=tiny
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ae_shards_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
DATA=$SCRATCH/data

cd "$REPO"
git pull --ff-only || echo "[probe] git pull failed, using existing checkout"

$PY -u scripts/probe_ae_shards.py \
  --ae "$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --sets \
    dclm:$DATA/cpt_cache_dclm \
    edu:$DATA/cpt_cache_edu \
    oldweb:$DATA/fineweb10b_v2_word \
  --n 3000

echo "[done] AE shard probe finished"
