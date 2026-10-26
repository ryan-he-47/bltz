#!/bin/bash
# bltz x lm-evaluation-harness battery (2026-10-26): CPT terminal ckpt vs p2
# baseline, 8 tasks, 0-shot, unit-space native scoring (raw acc cross-check), FULL split
# (--limit 1000000; adapter default is 32 and MUST be overridden).
# One ckpt per GPU, two in parallel on one node:
#   GPU0: word_cpt_s_anneal/last.pt   (S-tier CPT terminal, step 96291)
#   GPU1: word_05b_p2/last.pt         (p2 baseline, step 50863)
# Requires the venv + warmed HF cache from slurm/bltz_lmeval_setup.sbatch.
#
#   cd /gpfs1/home/yihe47/bltz/repo
#   sbatch slurm/bltz_lmeval.sbatch
#
#SBATCH --job-name=bltz_lmeval
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/lmeval_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
VENV=$SCRATCH/venv_lmeval
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
CPT=$SCRATCH/ckpts/word_cpt_s_anneal/last.pt
P2=$SCRATCH/ckpts/word_05b_p2/last.pt
LOGDIR=/gpfs1/home/yihe47/bltz/logs
TASKS=arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,commonsense_qa

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOGDIR"

echo "[job] host=$(hostname) job=$SLURM_JOB_ID date=$(date)"
echo "[job] TASKS=$TASKS"
echo "[job] HF_HOME=$HF_HOME"
echo "[job] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "[job] dataset cache:"; ls "$HF_DATASETS_CACHE" || echo "[warn] dataset cache missing"

# --- fail-fast environment checks ------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[err] venv missing at $VENV -- run slurm/bltz_lmeval_setup.sbatch first"
  exit 2
fi
"$VENV/bin/python" -c 'import lm_eval; print("[job] lm_eval", lm_eval.__version__)'
echo "[job] GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CPT_LOG=$LOGDIR/lmeval_cpt_s_${SLURM_JOB_ID}.log
P2_LOG=$LOGDIR/lmeval_p2_${SLURM_JOB_ID}.log
echo "[job] cpt log: $CPT_LOG"
echo "[job] p2  log: $P2_LOG"

# --- two ckpts, one GPU each, in parallel ---------------------------------
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u scripts/lmeval_bltz.py \
  --ckpt "$CPT" --ae "$AE" --tok "$TOK" --device cuda \
  --tasks "$TASKS" \
  --num-fewshot 0 --limit 1000000 --batch-size 1 \
  > "$CPT_LOG" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$VENV/bin/python" -u scripts/lmeval_bltz.py \
  --ckpt "$P2" --ae "$AE" --tok "$TOK" --device cuda \
  --tasks "$TASKS" \
  --num-fewshot 0 --limit 1000000 --batch-size 1 \
  > "$P2_LOG" 2>&1 &
PID1=$!

echo "[job] launched cpt pid=$PID0 (GPU0) p2 pid=$PID1 (GPU1)"
RC0=0; RC1=0
wait "$PID0" || RC0=$?
wait "$PID1" || RC1=$?
echo "[done] lm-eval battery exit=$((RC0 + RC1)) (cpt=$RC0 p2=$RC1) elapsed=${SECONDS}s"

for F in "$CPT_LOG" "$P2_LOG"; do
  echo "===== [results] $F ====="
  grep -A 200 '\[results\]' "$F" || echo "[warn] no [results] line in $F"
done
echo "[done] $(date)"
