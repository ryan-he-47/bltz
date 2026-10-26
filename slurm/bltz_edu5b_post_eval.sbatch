#!/bin/bash
# Post-training eval for the edu5b experiment (docs/50 §5.3, E1).
#   GPU0: mc_eval.py -> word_edu5b/last.pt (8-task unit-space battery)
#   GPU1: run_accept.py -> word_edu5b/last.pt (4-section internal battery;
#         includes diag_v2/snap-500k/semantic/gen_accept) then diag_v2 on the
#         edu held-out pkl + milestone NLL curve (old dist).
# Requires: $SCRATCH/data/edu_eval.pkl (created by the pre_eval job).
#   sbatch slurm/bltz_edu5b_post_eval.sbatch
#SBATCH --job-name=bltz_edu5b_post_eval
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-05:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/edu5b_post_eval_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
VENV=$SCRATCH/venv_lmeval
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
CKPT=$SCRATCH/ckpts/word_edu5b/last.pt
LOGDIR=/gpfs1/home/yihe47/bltz/logs
OUTDIR=$SCRATCH/mc_eval
TASKS=arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,commonsense_qa
OLDPKL=$SCRATCH/data/old_word_eval.pkl
EDUPKL=$SCRATCH/data/edu_eval.pkl
OLDCACHE=$SCRATCH/data/fineweb10b_v2_word

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOGDIR" "$OUTDIR"

[ -f "$CKPT" ] || { echo "[err] missing $CKPT"; exit 2; }
[ -f "$EDUPKL" ] || { echo "[err] missing $EDUPKL (run pre_eval first)"; exit 2; }
[ -x "$VENV/bin/python" ] || { echo "[err] venv missing at $VENV"; exit 2; }

CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u scripts/mc_eval.py \
  --ckpt "$CKPT" --ae "$AE" --tok "$TOK" --tag edu5b \
  --tasks "$TASKS" --limit 0 --pmi --out-dir "$OUTDIR" \
  > "$LOGDIR/edu5b_post_mc_${SLURM_JOB_ID}.log" 2>&1 &
P0=$!

(
  export CUDA_VISIBLE_DEVICES=1
  "$PY" -u scripts/run_accept.py "$CKPT" "$AE" --tag edu5b \
    > "$LOGDIR/edu5b_post_accept_${SLURM_JOB_ID}.log" 2>&1
  # edu in-domain NLL (fit, not a capability metric) + old-dist NLL
  "$PY" -u scripts/diag_v2.py "$CKPT" "$AE" --units-pkl "$EDUPKL" \
    > "$LOGDIR/edu5b_post_edunll_${SLURM_JOB_ID}.log" 2>&1 || true
  "$PY" -u scripts/diag_v2.py "$CKPT" "$AE" --units-pkl "$OLDPKL" \
    > "$LOGDIR/edu5b_post_oldnll_${SLURM_JOB_ID}.log" 2>&1 || true
) &
P1=$!

echo "[job] launched mc_eval pid=$P0 (GPU0), run_accept+diag pid=$P1 (GPU1)"
RC0=0; RC1=0
wait "$P0" || RC0=$?
wait "$P1" || RC1=$?
echo "[done] post_eval exit=$((RC0+RC1)) (mc=$RC0 internal=$RC1) elapsed=${SECONDS}s"

# milestone old-dist NLL curve (single GPU, sequential; optional but cheap)
for S in 50000 55000 60000 65000 70000; do
  F=$SCRATCH/ckpts/word_edu5b/ckpt_s$(printf '%07d' "$S").pt
  [ -f "$F" ] || continue
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/diag_v2.py "$F" "$AE" \
    --units-pkl "$OLDPKL" > "$LOGDIR/edu5b_mile_${S}_oldnll.log" 2>&1 || true
done
echo "[done] $(date)"
