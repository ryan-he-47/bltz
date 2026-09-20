#!/bin/bash
# AE full acceptance trio (docs/32; 2026-09-19 拍板: diag BEFORE POC).
# Runs on ae_32_256_full/best.pt (the 32-dim full AE — d_emb 配置事故意外产出,
# 用户拍板顺势先试水 32 维) against the v2 cache; output saved into the
# ckpt dir as diag.txt. POC submission stays MANUAL after reviewing it.
#   sbatch slurm/bltz_ae_diag.sbatch
#SBATCH --job-name=bltz_ae_diag
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ae_diag_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

cd "$REPO"
git pull --ff-only || echo "[diag] git pull failed, using existing checkout"

$PY -u scripts/diag_ae_v2.py "$SCRATCH/ckpts/ae_32_256_full/best.pt" \
  --cache "$SCRATCH/data/fineweb10b_v2" \
  2>&1 | tee "$SCRATCH/ckpts/ae_32_256_full/diag.txt"

echo "[done] AE acceptance diag; review diag.txt before submitting bltz_v2_60k"
