#!/bin/bash
# bltz x lm-evaluation-harness: isolated install + HF dataset warm-up (2026-10-26).
# CPU-only tiny-partition job (login-node discipline: pip install must NOT run on
# the login node). Creates a scratch venv with --system-site-packages (reuses the
# conda cose2 torch/datasets/transformers/... without touching that env), installs
# lm-eval==0.4.13 INSIDE the venv only, then warms the 8-task HF dataset cache in
# $SCRATCH/hf_cache so the GPU eval jobs can run from cache.
#
#   cd /gpfs1/home/yihe47/bltz/repo
#   sbatch --wait slurm/bltz_lmeval_setup.sbatch
#
#SBATCH --job-name=bltz_lmeval_setup
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/lmeval_setup_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
VENV=$SCRATCH/venv_lmeval
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
CKPT=$SCRATCH/ckpts/word_05b_p2/last.pt
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
TASKS=arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,commonsense_qa

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

cd "$REPO"
echo "[setup] host=$(hostname) job=$SLURM_JOB_ID date=$(date)"
echo "[setup] HF_HOME=$HF_HOME"
echo "[setup] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
$PY -V

# --- pollution guard: conda env package snapshot before/after ---------------
$PY -m pip freeze > "$SCRATCH/lmeval_setup_conda_before.txt" 2>&1 || true

# --- 1) isolated venv -------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv --system-site-packages $VENV"
  $PY -m venv --system-site-packages "$VENV"
else
  echo "[setup] venv already exists, reusing: $VENV"
fi

# --- 2) install lm-eval 0.4.13 (venv only) ---------------------------------
echo "[setup] pip install lm-eval==0.4.13"
"$VENV/bin/pip" install --no-input --disable-pip-version-check "lm-eval==0.4.13"

# --- 3) verify versions + install location ---------------------------------
"$VENV/bin/python" - <<'PYEOF'
import sys
import lm_eval, torch, datasets, transformers, accelerate, evaluate
print("[setup] python      :", sys.version.split()[0], sys.executable)
print("[setup] lm_eval     :", lm_eval.__version__, lm_eval.__file__)
print("[setup] torch       :", torch.__version__, torch.__file__)
print("[setup] datasets    :", datasets.__version__, datasets.__file__)
print("[setup] transformers:", transformers.__version__, transformers.__file__)
print("[setup] accelerate  :", accelerate.__version__)
print("[setup] evaluate    :", evaluate.__version__)
PYEOF

# --- 4) warm HF datasets with a CPU smoke run of the adapter ---------------
# 8 tasks x 1 example, unit_nll (cheap): downloads each task's HF dataset into
# $HF_DATASETS_CACHE so the GPU eval jobs can run cache-only.
WARM_LOG=$SCRATCH/lmeval_warm_all.log
echo "[setup] warm-up: $TASKS (log: $WARM_LOG)"
set +e
"$VENV/bin/python" -u scripts/lmeval_bltz.py \
  --ckpt "$CKPT" --ae "$AE" --tok "$TOK" --device cpu \
  --tasks "$TASKS" --num-fewshot 0 --limit 1 \
  > "$WARM_LOG" 2>&1
WRC=$?
set -e
if [ "$WRC" -eq 0 ]; then
  echo "[setup] warm-up ALL-8 OK"
else
  echo "[setup] warm-up ALL-8 failed rc=$WRC; retrying task-by-task to isolate"
  FAILED=""
  for T in arc_easy arc_challenge hellaswag piqa winogrande openbookqa boolq commonsense_qa; do
    if "$VENV/bin/python" -u scripts/lmeval_bltz.py \
         --ckpt "$CKPT" --ae "$AE" --tok "$TOK" --device cpu \
         --tasks "$T" --num-fewshot 0 --limit 1 \
         > "$SCRATCH/lmeval_warm_${T}.log" 2>&1; then
      echo "[setup] WARM OK   $T"
    else
      echo "[setup] WARM FAIL $T (see $SCRATCH/lmeval_warm_${T}.log)"
      FAILED="$FAILED $T"
    fi
  done
  if [ -n "$FAILED" ]; then echo "[setup] FAILED TASKS:$FAILED"; else echo "[setup] retry ALL OK"; fi
fi

echo "[setup] warm-up log tail (results should list all 8 tasks):"
tail -n 60 "$WARM_LOG" || true

# --- 5) pollution check: conda env pip freeze must be identical ------------
$PY -m pip freeze > "$SCRATCH/lmeval_setup_conda_after.txt" 2>&1 || true
if diff -q "$SCRATCH/lmeval_setup_conda_before.txt" "$SCRATCH/lmeval_setup_conda_after.txt" > /dev/null; then
  echo "[setup] CONDA ENV UNCHANGED (pip freeze identical)"
else
  echo "[setup] !! CONDA ENV CHANGED - see lmeval_setup_conda_before/after.txt"
  diff "$SCRATCH/lmeval_setup_conda_before.txt" "$SCRATCH/lmeval_setup_conda_after.txt" || true
fi

# --- 6) HF cache footprint ------------------------------------------------
echo "[setup] hf_cache size:"; du -sh "$SCRATCH/hf_cache" || true
echo "[setup] HF_DATASETS_CACHE listing:"; ls "$HF_DATASETS_CACHE" 2>/dev/null || true
echo "[setup] DONE $(date)"
