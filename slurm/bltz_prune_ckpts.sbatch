#!/bin/bash
# Scratch ckpt pruning (2026-10-26 quota: bltz scratch 276GB vs 300GB soft
# quota; SFT needs ~30GB). Deletions MUST run as a job (login node is
# read-only by house rule). Dry-run by default:
#
#   sbatch slurm/bltz_prune_ckpts.sbatch              # dry-run (safe)
#   sbatch slurm/bltz_prune_ckpts.sbatch --apply      # delete (after approval)
#SBATCH --job-name=bltz_prune
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/prune_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

echo "[prune] job $SLURM_JOB_ID start $(date '+%F %T') args=${*:-<dry-run>}"
cd "$REPO"
git pull --ff-only || echo "[prune] git pull skipped/failed"

du -sh "$SCRATCH/ckpts" "$SCRATCH/data" 2>/dev/null || true
$PY -u scripts/prune_ckpts.py "$@"
echo "[prune] after:"
du -sh "$SCRATCH" 2>/dev/null || true
echo "[prune] DONE $(date '+%F %T')"
