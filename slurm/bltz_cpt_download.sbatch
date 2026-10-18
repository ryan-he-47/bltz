#!/bin/bash
# CPT data acquisition (docs/43 §1, 2026-10-18): download DCLM-baseline slice /
# FineWeb-Edu slice (id-dedup vs sample-10BT) / FineMath+Stack-Edu anneal slices /
# smol-smoltalk, straight from HF to $SCRATCH/data. CPU-only; runs on tiny
# partition so no GPU is held; per-file heartbeats in the .out log.
#   sbatch slurm/bltz_cpt_download.sbatch
#SBATCH --job-name=bltz_cpt_dl
#SBATCH --partition=tiny
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cpt_dl_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

cd "$REPO"
git pull --ff-only || echo "[dl] git pull failed, using existing checkout"

$PY -u scripts/download_cpt_data.py --phase all --out "$SCRATCH/data"

echo "[done] CPT data download finished; log at $REPO/logs (see .out above)"
