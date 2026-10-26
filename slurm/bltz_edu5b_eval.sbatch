#!/bin/bash
# Post-training evaluation for the edu5b experiment (docs/50 §5.3).
#
# 2026-10-27 拍板: 评估预算延后到训练结束("等训完再说") -> E0(训练前 baseline,
# 含 CPT 对照)与 E1(训练后 endpoint)合并为**一个** 2-GPU 窗口,一次评三个 ckpt:
#   p2pre  = word_05b_p2/ckpt_pre_decay.pt       (分支 baseline, step 45429)
#   cpts7  = word_cpt_s_stable/ckpt_s0070000.pt   (同算力混合数据对照, +24571 步)
#   edu5b  = word_edu5b/last.pt                   (本实验 endpoint, +25432 步)
# 三者均须保留(prune 不执行; scripts/prune_ckpts.py 已把 s0070000 从删除清单剔除)。
#
#   CPU : dump_eval_pkl -> $SCRATCH/data/edu_eval.pkl (edu in-domain held-out)
#   GPU0: mc_eval.py 8-task unit-space battery x {p2pre, cpts7, edu5b}
#   GPU1: run_accept.py 4-section internal battery x {p2pre, cpts7, edu5b},
#         then diag_v2 edu/old NLL x 三个 ckpt, then edu5b 里程碑旧分布 NLL 曲线
#
# Outputs:
#   $SCRATCH/mc_eval/mc_eval_{p2pre,cpts7,edu5b}.{json,md}
#   <repo>/checkpoints/word_{p2pre,cpts7,edu5b}_accept.log (run_accept 默认产物)
#   $LOGDIR/edu5b_eval_*_<jobid>.log
#
#   sbatch slurm/bltz_edu5b_eval.sbatch
#SBATCH --job-name=bltz_edu5b_eval
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-10:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/edu5b_eval_%j.out

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
OLDPKL=$SCRATCH/data/fineweb10b_v2_word/eval_sample.pkl
EDUPKL=$SCRATCH/data/edu_eval.pkl
EDUCACHE=$SCRATCH/data/cpt_cache_edu

P2PRE=$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt
CPTS7=$SCRATCH/ckpts/word_cpt_s_stable/ckpt_s0070000.pt
EDU5B=$SCRATCH/ckpts/word_edu5b/last.pt

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOGDIR" "$OUTDIR"

for F in "$P2PRE" "$CPTS7" "$EDU5B"; do
  [ -f "$F" ] || { echo "[err] missing $F"; exit 2; }
done
[ -x "$VENV/bin/python" ] || { echo "[err] venv missing at $VENV"; exit 2; }

# ---- CPU: edu held-out eval pkl (once; cheap, only reads ~300 sampled seqs) ----
if [ ! -f "$EDUPKL" ]; then
  echo "[prep] dumping $EDUPKL"
  $PY -u scripts/dump_eval_pkl.py "$EDUCACHE" "$EDUPKL" \
    > "$LOGDIR/edu5b_eval_dump_${SLURM_JOB_ID}.log" 2>&1
fi

run_mc() {  # GPU0: unit-space 8-task battery
  local tag=$1 ck=$2
  echo "[mc] start $tag ($ck) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u scripts/mc_eval.py \
    --ckpt "$ck" --ae "$AE" --tok "$TOK" --tag "$tag" \
    --tasks "$TASKS" --limit 0 --pmi --out-dir "$OUTDIR" \
    >> "$LOGDIR/edu5b_eval_mc_${SLURM_JOB_ID}.log" 2>&1 \
    || echo "[mc] $tag FAILED rc=$?"
  echo "[mc] done $tag $(date '+%T')"
}

run_accept() {  # GPU1: 4-section internal battery (old-dist forgetting + semantics + gen)
  local tag=$1 ck=$2
  echo "[accept] start $tag ($ck) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/run_accept.py "$ck" "$AE" --tag "$tag" \
    >> "$LOGDIR/edu5b_eval_accept_${SLURM_JOB_ID}.log" 2>&1 \
    || echo "[accept] $tag FAILED rc=$?"
  echo "[accept] done $tag $(date '+%T')"
}

(
  run_mc p2pre "$P2PRE"
  run_mc cpts7 "$CPTS7"
  run_mc edu5b "$EDU5B"
) &
P0=$!

(
  run_accept p2pre "$P2PRE"
  run_accept cpts7 "$CPTS7"
  run_accept edu5b "$EDU5B"
  # in-domain edu NLL (fit, not a capability metric) + old-dist NLL (forgetting)
  for pair in "p2pre:$P2PRE" "cpts7:$CPTS7" "edu5b:$EDU5B"; do
    tag=${pair%%:*}; ck=${pair#*:}
    CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/diag_v2.py "$ck" "$AE" --units-pkl "$EDUPKL" \
      >> "$LOGDIR/edu5b_eval_edunll_${tag}_${SLURM_JOB_ID}.log" 2>&1 || true
    CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/diag_v2.py "$ck" "$AE" --units-pkl "$OLDPKL" \
      >> "$LOGDIR/edu5b_eval_oldnll_${tag}_${SLURM_JOB_ID}.log" 2>&1 || true
  done
  # edu5b milestone old-dist NLL curve (forgetting red line trajectory)
  for S in 50000 55000 60000 65000 70000; do
    F=$SCRATCH/ckpts/word_edu5b/ckpt_s$(printf '%07d' "$S").pt
    [ -f "$F" ] || continue
    CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/diag_v2.py "$F" "$AE" \
      --units-pkl "$OLDPKL" > "$LOGDIR/edu5b_mile_${S}_oldnll.log" 2>&1 || true
  done
) &
P1=$!

echo "[job] launched: GPU0 mc_eval pid=$P0, GPU1 accept pid=$P1"
RC0=0; RC1=0
wait "$P0" || RC0=$?
wait "$P1" || RC1=$?
echo "[done] eval exit=$((RC0+RC1)) (mc=$RC0 accept=$RC1) elapsed=${SECONDS}s $(date)"
