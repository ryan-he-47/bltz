#!/bin/bash
# SFT template / marker probe (docs/48 §2 前置验证; CPU-only tiny job 2026-10-26).
# Renders REAL smol-smoltalk rows with the proposed chat template (3 separator
# variants), segments with segment_word, and asks the frozen word-AE whether
# marker units reconstruct as well as content (verify flag rate).  Answers:
# marker OOD risk, cross-turn symbol merges, target ratio, text conservation.
#
#   sbatch slurm/bltz_sft_template.sbatch
#SBATCH --job-name=bltz_sft_template
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/sft_template_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
DATA=$SCRATCH/data/sft/smol-smoltalk/data
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt

echo "[tmpl] job $SLURM_JOB_ID start $(date '+%F %T') on $(hostname)"
cd "$REPO"
git pull --ff-only || echo "[tmpl] git pull skipped/failed"

$PY -u scripts/probe_sft_template.py \
  --parquet-dir "$DATA" \
  --tok "$TOK" \
  --ae "$AE" \
  --n 100 \
  --out "$SCRATCH/data/sft_template_probe.json"

echo "[tmpl] DONE $(date '+%F %T')"
