#!/bin/bash
# 50-step smoke for the FineWeb-Edu-only 5B continuation (docs/50 §3).
# Same config as slurm/bltz_word_edu5b.sbatch, separate ckpt dir, tiny budget:
# verifies branch resume (step=45429), data loader on cpt_cache_edu, first-step
# loss (smoothness baseline for the full run), step time, fp16 stability.
# Expected: 50 logged steps ~10.5-11.5 s/step; first-step loss = p2 branch loss
# on the Edu stream (un-updated, lr~4e-7).
#   sbatch slurm/bltz_word_edu5b_smoke.sbatch
#SBATCH --job-name=bltz_edu5b_smoke
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-00:40:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_edu5b_smoke_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
CKPT=$SCRATCH/ckpts/word_edu5b_smoke
CACHE=$SCRATCH/data/cpt_cache_edu
RESUME0=$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
rm -rf "$CKPT"  # smoke is disposable: always restart clean
mkdir -p "$CKPT"

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29542 \
  scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set data.cache_dir="$CACHE" \
  --set data.n_patches=4096 \
  --set data.rotate_shards.resident=2 --set data.rotate_shards.every=2000 \
  --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
  --set model.grad_ckpt=true --set model.attn_doc_mask=xformers \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.fp16_init_scale=1024.0 --set train.fp16_growth_interval=500 \
  --set train.rewarmup=500 \
  --set train.batch=24 --set train.lr=2.0e-4 --set train.warmup=2000 \
  --set train.steps=45479 --set train.decay_frac=0.0 --set train.final_frac=1.0 \
  --set train.log_every=10 \
  --set train.ckpt_every=25 --set train.ckpt_keep=1 \
  --set train.milestone_every=1000000 \
  --set train.ckpt_dir="$CKPT" \
  --set train.resume="$RESUME0"

echo "[smoke] DONE: inspect $CKPT/train.log (expect start step=45429, 50 steps)"
