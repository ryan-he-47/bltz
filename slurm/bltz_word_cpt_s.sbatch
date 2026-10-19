#!/bin/bash
# CPT S-tier continued pretraining (docs/43, 2026-10-18 拍板): 0.5B word arm,
# +10B units on NEW data (60% DCLM + 40% FineWeb-Edu-dedup, zero replay).
# Branch point: word_05b_p2/ckpt_pre_decay.pt (step 45429, weight-only).
# Two phases in ONE sbatch (mid-run data switch = two trainer invocations):
#   A word_cpt_s_stable : step 45429 -> 91205 (= +45776 步新数据), plateau LR
#                         2e-4 (decay_frac=0), cache = cpt_cache_main;
#                         weight-only + rewarmup 500.
#   B word_cpt_s_anneal : resume A ckpt_full (91205) -> 96292, decay tail
#                         decay_frac=0.0528 -> 0, cache = cpt_cache_anneal
#                         (as-built web/code/math = 88/5/7, docs/43 §3.4).
# Total NEW-data steps = 91205+96292-45429-91205... = 50863 steps x 196.6k
# units = 10.0B units; decay tail = 5087 步 = 10.0% of the new-data run.
# ⚠️ 步数全局口径(2026-10-19 事故修复): 即使 weight-only 加载, trainer 也
# 从 resume ckpt 恢复 step 计数器(p2 同款: 20004->50863)——train.steps 是
# 绝对终点不是新增量。初版误把 phase A 写成 0->45776, 被 train.log 起步
# step=45450 当场抓住, STOP 优雅停后修正为上述绝对步数。
# wall-out auto-resubmit (p2-proven recipe): watchdog touches STOP at
# wall-300s; after wait, "clean exit + no last.pt + near wall" -> resubmit
# THIS file; manual STOP does not trigger resubmit (not near wall); crash
# (non-zero) never resubmits.
#   sbatch slurm/bltz_word_cpt_s.sbatch
#SBATCH --job-name=bltz_word_cpt_s
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_cpt_s_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
DATA=$SCRATCH/data
CKPT_A=$SCRATCH/ckpts/word_cpt_s_stable
CKPT_B=$SCRATCH/ckpts/word_cpt_s_anneal
WALL_S=$((3 * 86400))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"

run_phase() {
  local CKPT=$1 CACHE=$2 STEPS=$3 DFRAC=$4 FFRAC=$5 REWARMUP=$6 RESUME0=$7
  mkdir -p "$CKPT"
  local RESUME=(--set "train.resume=$RESUME0")
  if [ -f "$CKPT/ckpt_full.pt" ]; then
    RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
    echo "[train] resuming own state from $CKPT/ckpt_full.pt"
  fi
  local REM=$((WALL_S - 300 - SECONDS)); [ $REM -lt 60 ] && REM=60
  ( sleep $REM; touch "$CKPT/STOP" ) &
  local WD=$!

  $PY -m torch.distributed.run --nproc_per_node=2 --master_port=29533 \
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
    ${RESUME[@]+"${RESUME[@]}"} &
  local MAIN=$!
  local EXIT_CODE=0
  wait $MAIN || EXIT_CODE=$?
  kill $WD 2>/dev/null || true

  if [ "$EXIT_CODE" -eq 0 ] && [ ! -f "$CKPT/last.pt" ] && [ "$SECONDS" -ge $((WALL_S - 900)) ]; then
    echo "[wall] near-wall graceful stop (elapsed ${SECONDS}s) -> resubmit"
    sbatch "$REPO/slurm/bltz_word_cpt_s.sbatch" || echo "[wall] resubmit FAILED - manual resubmit needed"
    exit 0
  fi
  if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[err] phase $CKPT exited $EXIT_CODE - NOT resubmitting"
    exit "$EXIT_CODE"
  fi
}

if [ ! -f "$CKPT_A/last.pt" ]; then
  echo "===== phase A: stable on cpt_cache_main (45429 -> 91205) ====="
  run_phase "$CKPT_A" "$DATA/cpt_cache_main" 91205 0.0 0.1 500 \
    "$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt"
fi

if [ ! -f "$CKPT_B/last.pt" ]; then
  echo "===== phase B: anneal tail on cpt_cache_anneal (91205 -> 96292) ====="
  run_phase "$CKPT_B" "$DATA/cpt_cache_anneal" 96292 0.0528 0.0 0 \
    "$CKPT_A/ckpt_full.pt"
fi

echo "[done] CPT S-tier complete (A +45776 + B +5087 = +50863 steps = 10.0B units)"
