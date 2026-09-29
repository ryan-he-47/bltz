#!/bin/bash
# word-AE 2.5B-patch full pretrain (2026-09-29 user recipe):
#   data : full word-arm cache, stream mode, 2.5B patches = 108500 x 23040
#   model: 48-256, inducing=0 (STANDARD self-attn, no inducing points),
#          dec_hidden=384, everything else unchanged
#   typo : p 0.05 (freq throttle on)
#   LR   : cosine, warmup 1000, peak 4e-4 -> floor 4e-6
#   numerics: fp16 + GradScaler + gn spike guard (main-run aligned);
#          batch 23040 = local memtest 1.085MB/unit -> 25.0GiB on V100
#          (22% headroom under the 6GB/20% rule)
#   recovery: auto-resume from last.pt (re-submit to continue).
#   sbatch slurm/bltz_ae_word_full.sbatch
#SBATCH --job-name=bltz_ae_word_full
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/ae_word_full_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/ae_48_256_word_full

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

$PY -u scripts/train_ae.py configs/ae.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.stream=true \
  --set typo.p=0.05 \
  --set model.inducing=0 --set model.dec_hidden=384 \
  --set train.batch=23040 --set train.steps=108500 \
  --set train.lr=4.0e-4 --set train.warmup=1000 \
  --set train.sched=cosine --set train.final=4.0e-6 \
  --set train.bf16=false --set train.ckpt_every=5000 \
  --set train.log_every=500 \
  --set train.ckpt_dir="$CKPT"

echo "[done] word-AE 2.5B pretrain finished; best at $CKPT/best.pt"
