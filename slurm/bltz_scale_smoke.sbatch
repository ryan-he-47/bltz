#!/bin/bash
# Scale smoke (2026-09-27 用户拍板: 0.5B + 全量 + 4k ctx + 2 卡):
# measure peak VRAM / step time / attention backend for the 0.5B config at
# S=4096 across per-GPU batch sizes -> decides batch/LR of the full run.
#   sbatch slurm/bltz_scale_smoke.sbatch
#SBATCH --job-name=bltz_scale_smoke
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/scale_smoke_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[smoke] git pull failed, using existing checkout"

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29517 scripts/smoke_scale.py configs/v2.yaml \
  --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
  --set model.d_emb=48 --set model.mdn_depth=0 --set model.grad_ckpt=true \
  --set data.n_patches=4096 --set data.cache_dir="$SCRATCH/data/fineweb10b_v2" \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_typo_full/best.pt" \
  --set train.bf16=false

echo "[smoke] done"
