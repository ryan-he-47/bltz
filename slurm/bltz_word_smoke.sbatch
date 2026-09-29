#!/bin/bash
# Word-arm formal-run smoke (2026-09-29): batch-VRAM/speed sweep on V100 for
# BOTH arms (frozen / learnable input) with online verify ON, fp16, S=512.
# Decides per-arm batch under the 4-6GB headroom rule.
#   sbatch slurm/bltz_word_smoke.sbatch
#SBATCH --job-name=bltz_word_smoke
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_smoke_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[smoke] git pull failed, using existing checkout"

COMMON="configs/v2_word.yaml
  --set data.cache_dir=$SCRATCH/data/fineweb10b_v2_word
  --set train.ae_ckpt=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
  --set train.verify.tok_dir=$TOK
  --set train.bf16=false"

echo "=== arm FROZEN ==="
SMOKE_BATCHES="32,64,80" $PY scripts/smoke_scale.py $COMMON \
  --set model.input_mode=frozen

echo "=== arm LEARNABLE ==="
SMOKE_BATCHES="24,32,36,40" $PY scripts/smoke_scale.py $COMMON \
  --set model.input_mode=learnable

echo "[smoke] done"
