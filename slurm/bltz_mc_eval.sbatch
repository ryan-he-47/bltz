#!/bin/bash
# bltz unit-space MC battery (docs/49, 2026-10-27): p2 vs CPT terminal ckpt,
# 8 tasks, 0-shot, FULL split, unit-space GMM scoring (acc_unit primary;
# acc_raw/acc_char/acc_unit_pmi reported too). Replaces the retired pushforward
# battery. One ckpt per GPU, two in parallel:
#   GPU0: word_cpt_s_anneal/last.pt   (S-tier CPT terminal, step 96292)
#   GPU1: word_05b_p2/last.pt         (p2 baseline, step 50863)
# Requires venv + warmed HF dataset cache (slurm/bltz_lmeval_setup.sbatch).
#
#   cd /gpfs1/home/yihe47/bltz/repo
#   sbatch slurm/bltz_mc_eval.sbatch
#
#SBATCH --job-name=bltz_mc_eval
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/mc_eval_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
VENV=$SCRATCH/venv_lmeval
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
CPT=$SCRATCH/ckpts/word_cpt_s_anneal/last.pt
P2=$SCRATCH/ckpts/word_05b_p2/last.pt
LOGDIR=/gpfs1/home/yihe47/bltz/logs
OUTDIR=$SCRATCH/mc_eval
TASKS=arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,commonsense_qa

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOGDIR" "$OUTDIR"

echo "[job] host=$(hostname) job=$SLURM_JOB_ID date=$(date)"
echo "[job] TASKS=$TASKS"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[err] venv missing at $VENV"; exit 2
fi
"$VENV/bin/python" -c 'import lm_eval, torch; print("[job] lm_eval", lm_eval.__version__, "| torch", torch.__version__)'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CPT_LOG=$LOGDIR/mc_eval_cpt_s_${SLURM_JOB_ID}.log
P2_LOG=$LOGDIR/mc_eval_p2_${SLURM_JOB_ID}.log
echo "[job] cpt log: $CPT_LOG"
echo "[job] p2  log: $P2_LOG"

CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u scripts/mc_eval.py \
  --ckpt "$CPT" --ae "$AE" --tok "$TOK" --tag cpt_s \
  --tasks "$TASKS" --limit 0 --pmi --out-dir "$OUTDIR" \
  > "$CPT_LOG" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$VENV/bin/python" -u scripts/mc_eval.py \
  --ckpt "$P2" --ae "$AE" --tok "$TOK" --tag p2 \
  --tasks "$TASKS" --limit 0 --pmi --out-dir "$OUTDIR" \
  > "$P2_LOG" 2>&1 &
PID1=$!

echo "[job] launched cpt pid=$PID0 (GPU0) p2 pid=$PID1 (GPU1)"
RC0=0; RC1=0
wait "$PID0" || RC0=$?
wait "$PID1" || RC1=$?
echo "[done] mc_eval exit=$((RC0 + RC1)) (cpt=$RC0 p2=$RC1) elapsed=${SECONDS}s"
for F in "$CPT_LOG" "$P2_LOG"; do
  echo "===== [summary] $F ====="
  grep -A 3 'SUMMARY' "$F" || echo "[warn] no SUMMARY in $F"
done
echo "[done] $(date)"
