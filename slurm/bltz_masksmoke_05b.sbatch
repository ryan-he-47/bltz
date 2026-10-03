#!/bin/bash
# 边界掩码 smoke (2026-10-03): production 形状(0.5B d1024/L28/S=4096/batch24,
# 核验+轮换 resident2) 跑 6 步; stingy 分区 (gpu-v100s-01).
# 2026-10-03 晚更新: xformers BlockDiagonalCausalMask 后端显式启用
# (attn_doc_mask=xformers, fail-loud), 测效率优化后端。
#   sbatch slurm/bltz_masksmoke_05b.sbatch
#SBATCH --job-name=bltz_mask_smoke
#SBATCH --partition=stingy
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/mask_smoke_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
CKPT=$SCRATCH/ckpts/word_05b_xf_smoke

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export BLTZ_MEM=1
export BLTZ_PROF=1

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29537 \
  scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.n_patches=4096 \
  --set data.rotate_shards.resident=2 --set data.rotate_shards.every=2000 \
  --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
  --set model.grad_ckpt=true --set model.attn_doc_mask=xformers \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.fp16_init_scale=1024.0 --set train.fp16_growth_interval=500 \
  --set train.batch=24 --set train.lr=2.0e-4 --set train.warmup=2000 \
  --set train.steps=6 --set train.decay_frac=0.2134 --set train.final_frac=0.0 \
  --set train.ckpt_every=250 --set train.ckpt_keep=3 \
  --set train.milestone_every=0 \
  --set train.ckpt_dir="$CKPT"

echo "[done] xf mask smoke finished; log at $CKPT/train.log"

