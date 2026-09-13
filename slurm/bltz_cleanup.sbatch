#!/bin/bash
# one-off scratch cleanup (2026-09-13 quota lesson: user scratch = 300GB, NOT
# the cluster-wide 787T). Login-node deletions are forbidden by house rule, so
# cleanup runs as a job.
#   sbatch slurm/bltz_cleanup.sbatch
#SBATCH --job-name=bltz_clean
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/clean_%j.out

set -euo pipefail
SCRATCH=/gpfs1/scratch/yihe47/bltz

# hf_cache (27GB): raw parquet downloads, re-downloadable in ~2h if a rebuild is
# ever needed; the built caches (data/, 85GB) are the product.
rm -rf "$SCRATCH/hf_cache"

# lr_probe_1e3 accident dir: keep frozen.pt (sole surviving stable-phase
# snapshot @step ~12000); delete the 24s no-op run's junk artifacts.
rm -f "$SCRATCH/ckpts/lr_probe_1e3/ckpt_full.pt" "$SCRATCH/ckpts/lr_probe_1e3/last.pt"

echo "[clean] done; usage now:"
du -sh "$SCRATCH"
