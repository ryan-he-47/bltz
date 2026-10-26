#!/bin/bash
# SFT data recon for smol-smoltalk (docs/48 施工方案前置测量; CPU-only tiny job).
# Reads the HF parquet shards, renders the proposed chat template, measures
# chars / Qwen-BPE tokens / word-arm units / assistant-target units /
# truncation@4096, and extrapolates total units -> steps & hours per epoch.
#
#   sbatch slurm/bltz_sft_probe.sbatch
#SBATCH --job-name=bltz_sft_probe
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/sft_probe_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
DATA=$SCRATCH/data/sft/smol-smoltalk/data
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

echo "[probe] job $SLURM_JOB_ID start $(date '+%F %T') on $(hostname)"
cd "$REPO"
git pull --ff-only || echo "[probe] git pull skipped/failed, using existing checkout"
echo "[probe] HEAD=$(git rev-parse --short HEAD)"
ls -l "$DATA" | head -8

$PY -u scripts/probe_sft_data.py \
  --data-dir "$DATA" \
  --tok "$TOK" \
  --n 5000 \
  --out "$SCRATCH/data/sft_probe.json"

echo "[probe] DONE $(date '+%F %T')"
