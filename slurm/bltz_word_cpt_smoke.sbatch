#!/bin/bash
# CPT S-tier smoke (docs/43 §7): production-shape 3-step smoke of BOTH new
# caches before the formal run — A shape on cpt_cache_main (weight-only load
# from ckpt_pre_decay) + B shape on cpt_cache_anneal (resume from A's smoke
# ckpt_full). stingy partition (gpu-v100s-01, 5h limit, 冒烟专用).
# Verifies: ShardRotator on new shard geometry, verify/re-tokenize path,
# xformers doc-mask, ckpt save/load, fp16 scaler.
#   sbatch --dependency=afterok:<cachebuild_job> slurm/bltz_word_cpt_smoke.sbatch
#SBATCH --job-name=bltz_cpt_smoke
#SBATCH --partition=stingy
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_cpt_smoke_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
DATA=$SCRATCH/data
CKPT_A=$SCRATCH/ckpts/word_cpt_smoke_a
CKPT_B=$SCRATCH/ckpts/word_cpt_smoke_b

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[smoke] git pull failed, using existing checkout"

run_smoke() {
  local CKPT=$1 CACHE=$2 RESUME0=$3
  rm -rf "$CKPT"; mkdir -p "$CKPT"
  $PY -m torch.distributed.run --nproc_per_node=2 --master_port=29534 \
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
    --set train.steps=3 --set train.decay_frac=0.0 \
    --set train.ckpt_every=2 --set train.ckpt_keep=1 \
    --set train.milestone_every=0 \
    --set train.ckpt_dir="$CKPT" \
    --set "train.resume=$RESUME0"
}

echo "===== smoke A: cpt_cache_main (weight-only branch load) ====="
run_smoke "$CKPT_A" "$DATA/cpt_cache_main" "$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt"

echo "===== smoke B: cpt_cache_anneal (full-state resume from smoke A) ====="
run_smoke "$CKPT_B" "$DATA/cpt_cache_anneal" "$CKPT_A/ckpt_full.pt"

rm -rf "$CKPT_A" "$CKPT_B"
echo "[done] CPT smoke passed on both caches"
