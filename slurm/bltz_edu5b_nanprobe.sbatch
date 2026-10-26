#!/bin/bash
# Round 2 NaN triage (2026-10-27). Round 1 showed the FULL-PRELOAD reader is
# clean for 8 steps on all three caches, so the suspect is the ShardRotator
# path (resident=2, rotate every 2000) that the smoke used:
#   smoke 693674: step 45430 finite (loss -40.0819, gn 19.7); steps 45431+
#   all gn=NaN (persistent). Round-1 probe used preload (no rotate) and got
#   DIFFERENT first-batch losses (edu -39.4722), i.e. a different data stream.
# This probe re-runs the EXACT smoke command line (rotate on, ckpt_every=25,
# ckpt_keep=1, milestone_every=1e6, steps 45438, log_every=1) on:
#   A) edu cache   -> expect: reproduces -40.0819 at step 45430 if deterministic
#   B) main cache  -> rotator on the CPT-known-good cache (control)
#   sbatch slurm/bltz_edu5b_nanprobe.sbatch
#SBATCH --job-name=bltz_nanprobe2
#SBATCH --partition=stingy
#SBATCH --nodelist=gpu-v100s-01
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/edu5b_nanprobe2_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
DATA=$SCRATCH/data
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
RESUME0=$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"

run_one() {  # TAG CACHE CKPT PORT
  local TAG=$1 CACHE=$2 CKPT=$3 PORT=$4
  echo "===== PROBE2 $TAG cache=$CACHE $(date '+%T') ====="
  mkdir -p "$CKPT"
  rm -f "$CKPT/STOP"
  $PY -m torch.distributed.run --nproc_per_node=2 --master_port=$PORT \
    scripts/train_bltz_v2.py configs/v2_word.yaml \
    --set data.cache_dir="$CACHE" \
    --set data.n_patches=4096 \
    --set data.rotate_shards.resident=2 --set data.rotate_shards.every=2000 \
    --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
    --set model.grad_ckpt=true --set model.attn_doc_mask=xformers \
    --set train.ae_ckpt="$AE" \
    --set train.verify.tok_dir="$TOK" \
    --set train.bf16=false \
    --set train.fp16_init_scale=1024.0 --set train.fp16_growth_interval=500 \
    --set train.rewarmup=500 \
    --set train.batch=24 --set train.lr=2.0e-4 --set train.warmup=2000 \
    --set train.steps=45438 --set train.decay_frac=0.0 --set train.final_frac=1.0 \
    --set train.log_every=1 \
    --set train.ckpt_every=25 --set train.ckpt_keep=1 \
    --set train.milestone_every=1000000 \
    --set train.ckpt_dir="$CKPT" \
    --set train.resume="$RESUME0" || echo "[probe2] $TAG run exited rc=$?"
  echo "===== PROBE2 $TAG done $(date '+%T'); log: $CKPT/train.log ====="
  tail -12 "$CKPT/train.log" 2>/dev/null || true
}

run_one edu_rot "$DATA/cpt_cache_edu" "$SCRATCH/ckpts/nanprobe2_edu" 29561
run_one main_rot "$DATA/cpt_cache_main" "$SCRATCH/ckpts/nanprobe2_main" 29562
echo "[probe2] ALL DONE $(date)"
