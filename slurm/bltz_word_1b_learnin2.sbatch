#!/bin/bash
# LEARNABLE arm RESTART after the fp16-scale cascade (2026-09-30 病因终局):
# resume from the HEALTHY milestone ckpt_s0002000 (pre-collapse basin), with
# fp16 hardening proven by the replay forensics:
#   * fp16_init_scale=1024 (replay showed scale >=8192 overflows the sharpened
#     density model's backward; <=1024 ran clean for 3000+ steps)
#   * fp16_growth_interval=500 (fast recovery if an event still occurs)
#   * rewarmup=500 (weight-only resume resets Adam; ramp LR back to peak over
#     500 steps — peak stays 4e-4, house rule peak<=parent-end-LR holds)
#   * ckpt_keep=3, no permanent milestones (2026-09-30 用户拍板, 硬盘额度)
# steps=67832 absolute (WSD shape preserved); remaining ~65.8k steps ~15h.
#   sbatch slurm/bltz_word_1b_learnin2.sbatch
#SBATCH --job-name=bltz_word_1b_learnin2
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_1b_learnin2_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[learnin2] git pull failed, using existing checkout"

$PY -u scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set model.input_mode=learnable \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.preload=true \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.fp16_init_scale=1024.0 \
  --set train.fp16_growth_interval=500 \
  --set train.rewarmup=500 \
  --set train.batch=36 \
  --set train.steps=67832 \
  --set train.ckpt_keep=3 \
  --set train.resume="$SCRATCH/ckpts/v2_word_1b_learnin/ckpt_s0002000.pt" \
  --set train.ckpt_dir="$SCRATCH/ckpts/v2_word_1b_learnin2"

echo "[learnin2] done"
