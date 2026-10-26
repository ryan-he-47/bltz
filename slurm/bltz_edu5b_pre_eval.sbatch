#!/bin/bash
# Pre-training baseline eval for the edu5b experiment (docs/50 §5.3, E0).
# MUST run BEFORE the prune job: reads CPT stable milestone ckpt_s0070000.pt
# (matched-compute mixed-data comparator; the prune manifest deletes it).
#   GPU0: mc_eval.py -> p2 ckpt_pre_decay (branch baseline, new unit-space metric)
#   GPU1: mc_eval.py -> CPT ckpt_s0070000 (+24,571 steps mixed data)
#   CPU : dump_eval_pkl.py on cpt_cache_edu -> $SCRATCH/data/edu_eval.pkl
# Requires the lm-eval venv (docs/49); HF datasets are cached on scratch.
#   sbatch slurm/bltz_edu5b_pre_eval.sbatch
#SBATCH --job-name=bltz_edu5b_pre_eval
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/edu5b_pre_eval_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
VENV=$SCRATCH/venv_lmeval
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
LOGDIR=/gpfs1/home/yihe47/bltz/logs
OUTDIR=$SCRATCH/mc_eval
TASKS=arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,commonsense_qa
P2PRE=$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt
CPTS7=$SCRATCH/ckpts/word_cpt_s_stable/ckpt_s0070000.pt

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOGDIR" "$OUTDIR"

[ -f "$P2PRE" ] || { echo "[err] missing $P2PRE"; exit 2; }
[ -f "$CPTS7" ] || { echo "[err] missing $CPTS7 (run BEFORE prune!)"; exit 2; }
[ -x "$VENV/bin/python" ] || { echo "[err] venv missing at $VENV"; exit 2; }

# --- 0) edu held-out eval pkl (CPU) ---------------------------------------
if [ ! -f "$SCRATCH/data/edu_eval.pkl" ]; then
  echo "[job] dumping edu eval pkl ..."
  "$PY" -u scripts/dump_eval_pkl.py "$SCRATCH/data/cpt_cache_edu" \
    "$SCRATCH/data/edu_eval.pkl"
else
  echo "[job] edu_eval.pkl exists, skip dump"
fi

# --- 1) two MC batteries in parallel (one ckpt per GPU) -------------------
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u scripts/mc_eval.py \
  --ckpt "$P2PRE" --ae "$AE" --tok "$TOK" --tag p2pre \
  --tasks "$TASKS" --limit 0 --pmi --out-dir "$OUTDIR" \
  > "$LOGDIR/edu5b_pre_mc_p2pre_${SLURM_JOB_ID}.log" 2>&1 &
P0=$!

CUDA_VISIBLE_DEVICES=1 "$VENV/bin/python" -u scripts/mc_eval.py \
  --ckpt "$CPTS7" --ae "$AE" --tok "$TOK" --tag cpt_s7 \
  --tasks "$TASKS" --limit 0 --pmi --out-dir "$OUTDIR" \
  > "$LOGDIR/edu5b_pre_mc_cpts7_${SLURM_JOB_ID}.log" 2>&1 &
P1=$!

echo "[job] launched mc_eval p2pre pid=$P0 (GPU0), cpt_s7 pid=$P1 (GPU1)"
RC0=0; RC1=0
wait "$P0" || RC0=$?
wait "$P1" || RC1=$?
echo "[done] pre_eval exit=$((RC0+RC1)) (p2pre=$RC0 cpt_s7=$RC1) elapsed=${SECONDS}s"
ls -la "$OUTDIR"/mc_eval_{p2pre,cpt_s7}.json 2>/dev/null || true
