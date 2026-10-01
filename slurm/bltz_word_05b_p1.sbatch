#!/bin/bash
# word-arm 0.5B scaling, PHASE 1/2 (2026-09-30 用户拍板): 5B units 验货段.
# Recipe = 0.5B scaling 原配方(2x V100 DDP, per-GPU batch 24, d1024/L28/H16,
# S=4096, grad_ckpt, LR 2e-4 flat, warmup 2000)+ word 臂两件套(在线核验
# 重分词 + 块级轮换 shard). 退火尾起于里程碑 20000(decay_frac 0.2134),
# final_frac=0 充分退火供验货;后半段从平台期里程碑 ckpt_s0020000 接续.
# fp16 硬化(init_scale 1024 / growth 500, 09-30 事故家规).
#   sbatch slurm/bltz_word_05b_p1.sbatch
#SBATCH --job-name=bltz_word_05b_p1
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_05b_p1_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
CKPT=$SCRATCH/ckpts/word_05b_p1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=()
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own state from $CKPT/ckpt_full.pt"
fi

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29531 \
  scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.n_patches=4096 \
  --set data.rotate_shards.resident=2 --set data.rotate_shards.every=2000 \
  --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
  --set model.grad_ckpt=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.fp16_init_scale=1024.0 --set train.fp16_growth_interval=500 \
  --set train.batch=24 --set train.lr=2.0e-4 --set train.warmup=2000 \
  --set train.steps=25432 --set train.decay_frac=0.2134 --set train.final_frac=0.0 \
  --set train.ckpt_every=250 --set train.ckpt_keep=3 \
  --set train.milestone_every=10000 \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"}

echo "[done] word-0.5B phase-1 (5B units) finished; log at $CKPT/train.log"
