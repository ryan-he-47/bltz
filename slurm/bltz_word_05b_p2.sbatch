#!/bin/bash
# word-arm 0.5B scaling, PHASE 2/2 (2026-09-30 用户拍板): 平台期接续段.
# 验货通过后才提交! 从 phase-1 平台期里程碑 ckpt_s0020000(=退火起点)
# 续训: 绝对步 20000 -> 50863(累计 10B units), 常量段 20000->45431,
# 退火尾 5432 步与 phase-1 同长(decay_frac 0.1068), final_frac=0.
# rewarmup 500(weight-only 重启 Adam 防护; peak 2e-4 = 平台期 LR, 家规满足).
#   sbatch slurm/bltz_word_05b_p2.sbatch   # 验货通过后
#SBATCH --job-name=bltz_word_05b_p2
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_05b_p2_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
CKPT=$SCRATCH/ckpts/word_05b_p2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29532 \
  scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.n_patches=4096 \
  --set data.rotate_shards.resident=4 --set data.rotate_shards.every=2000 \
  --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
  --set model.grad_ckpt=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.fp16_init_scale=1024.0 --set train.fp16_growth_interval=500 \
  --set train.rewarmup=500 \
  --set train.batch=24 --set train.lr=2.0e-4 --set train.warmup=2000 \
  --set train.steps=50863 --set train.decay_frac=0.1068 --set train.final_frac=0.0 \
  --set train.ckpt_every=250 --set train.ckpt_keep=3 \
  --set train.milestone_every=5000 \
  --set train.resume="$SCRATCH/ckpts/word_05b_p1/ckpt_s0020000.pt" \
  --set train.ckpt_dir="$CKPT"

echo "[done] word-0.5B phase-2 (to 10B units) finished; log at $CKPT/train.log"
