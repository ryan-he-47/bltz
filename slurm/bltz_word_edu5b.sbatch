#!/bin/bash
# FineWeb-Edu-only 5B-unit continuation from the p2 stable-segment end
# (docs/50, 2026-10-27;待拍板 experiment). Single data source = CPT's
# FineWeb-Edu cache, no dataset mixing, no code/math anneal.
# Branch: word_05b_p2/ckpt_pre_decay.pt (step 45429, weight-only;
#         = LR 2e-4 plateau point -> no LR jump, unlike p2 last@50863).
# Budget: +25432 steps x 196,608 units/step = 5.000B units
#         -> absolute end step 70861; last 2543 steps (10% of new data)
#         linear decay to lr=0 (same tail shape as CPT).
# Notes:
#   * train.steps is the ABSOLUTE end step (trainer restores step from the
#     resume ckpt even for weight-only loads; docs/39 #42 事故口径).
#   * weight-only resume resets Adam -> train.rewarmup=500 (CPT-proven).
#   * data: $SCRATCH/data/cpt_cache_edu = 5.349B units / 7 shards (id-deduped
#     FineWeb-Edu), 5.000B units = 93.5% single pass, zero replay.
# wall-out auto-resubmit: watchdog touches STOP at wall-300s; "clean exit +
# no last.pt + near wall" -> resubmit this file; manual STOP / crash never
# resubmits (p2/CPT-proven recipe, docs/39 #42 addendum).
#   sbatch slurm/bltz_word_edu5b.sbatch
#SBATCH --job-name=bltz_word_edu5b
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_edu5b_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
DATA=$SCRATCH/data
CKPT=$SCRATCH/ckpts/word_edu5b
CACHE=$DATA/cpt_cache_edu
RESUME0=$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt
WALL_S=$((3 * 86400))
STEPS=70861             # 45429 (branch) + 25432 (= 5.000B units)
DFRAC=0.035887          # decay tail = 2543 steps (10% of new data)
FFRAC=0.0               # decay to lr=0
REWARMUP=500

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"

trap 'echo "[sig] SIGTERM at elapsed ${SECONDS}s (wall)"; :' TERM

if [ ! -f "$RESUME0" ]; then
  echo "[err] branch ckpt missing: $RESUME0"; exit 2
fi
if [ ! -d "$CACHE" ]; then
  echo "[err] data cache missing: $CACHE"; exit 2
fi

mkdir -p "$CKPT"
rm -f "$CKPT/STOP"  # clear stale STOP from a previous segment

RESUME=(--set "train.resume=$RESUME0")
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own state from $CKPT/ckpt_full.pt"
fi

REM=$((WALL_S - 300 - SECONDS))
if [ $REM -lt 7200 ]; then
  echo "[wall] only ${REM}s wall left; NOT starting the phase -> resubmit"
  sbatch "$REPO/slurm/bltz_word_edu5b.sbatch" || echo "[wall] resubmit FAILED - manual resubmit needed"
  exit 0
fi
( sleep $REM; touch "$CKPT/STOP" ) &
WD=$!

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29541 \
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
  --set train.rewarmup=$REWARMUP \
  --set train.batch=24 --set train.lr=2.0e-4 --set train.warmup=2000 \
  --set train.steps=$STEPS --set train.decay_frac=$DFRAC --set train.final_frac=$FFRAC \
  --set train.ckpt_every=250 --set train.ckpt_keep=3 \
  --set train.milestone_every=5000 \
  --set train.ckpt_dir="$CKPT" \
  "${RESUME[@]}" &
MAIN=$!
EXIT_CODE=0
wait $MAIN || EXIT_CODE=$?
kill $WD 2>/dev/null || true

if [ ! -f "$CKPT/last.pt" ] && [ "$SECONDS" -ge $((WALL_S - 1800)) ]; then
  echo "[wall] near-wall exit (code $EXIT_CODE, elapsed ${SECONDS}s) -> resubmit"
  sbatch "$REPO/slurm/bltz_word_edu5b.sbatch" || echo "[wall] resubmit FAILED - manual resubmit needed"
  exit 0
fi
if [ "$EXIT_CODE" -ne 0 ]; then
  echo "[err] run exited $EXIT_CODE - NOT resubmitting"
  exit "$EXIT_CODE"
fi

echo "[done] word_edu5b complete: +25432 steps = 5.000B units, end step $STEPS"
