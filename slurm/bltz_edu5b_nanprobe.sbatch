#!/bin/bash
# NaN triage for the edu5b smoke (2026-10-27): with the p2 branch ckpt,
# step 45430 had finite loss/gn (-40.08 / 19.7) but every step from 45431 on
# produced gn=NaN (spike_skip). This probe runs 8 steps twice in ONE job:
#   A) cache = cpt_cache_edu   (7 shards, the new experiment's data)
#   B) cache = cpt_cache_main  (76 mixed symlink shards = CPT stable phase,
#                               known-good reference)
# log_every=1 so finite steps show loss/gn; the first NaN step is visible via
# the spike_skip lines. Deterministic repro at the same step => data/config bug;
# both caches identical => environment/node (e.g. bad stingy node) suspicion.
#   sbatch slurm/bltz_edu5b_nanprobe.sbatch
#SBATCH --job-name=bltz_nanprobe
#SBATCH --partition=stingy
#SBATCH --nodelist=gpu-v100s-01
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/edu5b_nanprobe_%j.out

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

run_one() {
  local TAG=$1 CACHE=$2 CKPT=$3 PORT=$4
  echo "===== PROBE $TAG cache=$CACHE $(date '+%T') ====="
  mkdir -p "$CKPT"
  rm -f "$CKPT/STOP"
  $PY -m torch.distributed.run --nproc_per_node=2 --master_port=$PORT \
    scripts/train_bltz_v2.py configs/v2_word.yaml \
    --set data.cache_dir="$CACHE" \
    --set data.n_patches=4096 \
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
    --set train.ckpt_every=100000 --set train.milestone_every=0 \
    --set train.ckpt_dir="$CKPT" \
    --set train.resume="$RESUME0" || echo "[probe] $TAG run exited rc=$?"
  echo "===== PROBE $TAG done $(date '+%T'); log: $CKPT/train.log ====="
  tail -12 "$CKPT/train.log" 2>/dev/null || true
}

run_one edu "$DATA/cpt_cache_edu" "$SCRATCH/ckpts/nanprobe_edu" 29551
run_one main "$DATA/cpt_cache_main" "$SCRATCH/ckpts/nanprobe_main" 29552
run_one p2data "$DATA/fineweb10b_v2_word" "$SCRATCH/ckpts/nanprobe_p2data" 29553
echo "[probe] ALL DONE $(date)"
