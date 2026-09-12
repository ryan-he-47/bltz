#!/bin/bash
# bltz Stage 1 - V100 calibration probe: real step time for both arms at
# batch 32 fp16 before committing a 5-day training job. Submit with:
#   sbatch --dependency=afterok:<cache_jobid> bltz_bench.sbatch
#SBATCH --job-name=bltz_bench
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/bench_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

cd "$REPO"
git pull --ff-only || echo "[bench] git pull failed, using existing checkout"

echo "=== bltz arm, batch 8/16/32, fp16 ==="
for B in 8 16 32; do
  $PY -u scripts/bench_twin.py \
    --set data.cache_dir="$SCRATCH/data/fineweb10b" \
    --set train.bf16=false --set train.batch=$B || echo "[bench] batch $B failed (OOM?)"
done

echo "=== token baseline arm, batch 32, fp16 ==="
$PY -u scripts/train_token_baseline.py configs/baseline.yaml \
  --set data.cache_dir="$SCRATCH/data/tokengpt2" \
  --set train.bf16=false --set train.batch=32 \
  --set train.steps=12 --set train.log_every=4 \
  --set train.ckpt_dir=$SCRATCH/ckpts/bench_baseline || echo "[bench] baseline failed"
rm -rf $SCRATCH/ckpts/bench_baseline

echo "[done] bench complete"
