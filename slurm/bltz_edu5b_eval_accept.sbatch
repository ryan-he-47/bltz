#!/bin/bash
# edu5b 训后评估 · 内部四段 + diag NLL + 里程碑曲线(单卡,2026-10-30 拆分版)。
# 与 slurm/bltz_edu5b_eval_mc.sbatch 配对;MC 主指标那份先跑,本份抢第二张空闲卡。
# 需要旧词协议资产(2026-10-30 从本机上传):
#   $SCRATCH/data/fineweb_word_local/ + $SCRATCH/protocol/protocol_table_500k.pt
# 集群策略:1 GPU 最多 8 CPU(10 会被 sbatch 拒),故 cpus-per-task=8。
#   sbatch slurm/bltz_edu5b_eval_accept.sbatch
#SBATCH --job-name=bltz_e5_acc
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/edu5b_eval_accept_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
LOGDIR=/gpfs1/home/yihe47/bltz/logs
OLDPKL=$SCRATCH/data/fineweb10b_v2_word/eval_sample.pkl
EDUPKL=$SCRATCH/data/edu_eval.pkl
EDUCACHE=$SCRATCH/data/cpt_cache_edu
OLDCACHE=$SCRATCH/data/fineweb_word_local

P2PRE=$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt
CPTS7=$SCRATCH/ckpts/word_cpt_s_stable/ckpt_s0070000.pt
EDU5B=$SCRATCH/ckpts/word_edu5b/last.pt

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOGDIR" checkpoints
[ -e data ] || ln -s "$SCRATCH/data" data
[ -e checkpoints/protocol_table_500k.pt ] || \
  ln -s "$SCRATCH/protocol/protocol_table_500k.pt" checkpoints/protocol_table_500k.pt

for F in "$P2PRE" "$CPTS7" "$EDU5B"; do
  [ -f "$F" ] || { echo "[err] missing $F"; exit 2; }
done

if [ ! -f "$EDUPKL" ]; then
  echo "[prep] dumping $EDUPKL"
  $PY -u scripts/dump_eval_pkl.py "$EDUCACHE" "$EDUPKL" \
    > "$LOGDIR/edu5b_eval_dump_${SLURM_JOB_ID}.log" 2>&1
fi

echo "[job] host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | tr '\n' '|')"

run_accept() {
  local tag=$1 ck=$2
  echo "[accept] start $tag $(date '+%T')"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/run_accept.py "$ck" "$AE" --tag "$tag" \
    > "$LOGDIR/edu5b_eval_accept_${tag}_${SLURM_JOB_ID}.log" 2>&1 \
    || echo "[accept] $tag FAILED rc=$?"
  echo "[accept] done $tag $(date '+%T')"
}

run_accept edu5b "$EDU5B"
run_accept p2pre "$P2PRE"
run_accept cpts7 "$CPTS7"

for pair in "p2pre:$P2PRE" "cpts7:$CPTS7" "edu5b:$EDU5B"; do
  tag=${pair%%:*}; ck=${pair#*:}
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/diag_v2.py "$ck" "$AE" --units-pkl "$EDUPKL" \
    > "$LOGDIR/edu5b_eval_edunll_${tag}_${SLURM_JOB_ID}.log" 2>&1 || true
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/diag_v2.py "$ck" "$AE" --units-pkl "$OLDPKL" \
    > "$LOGDIR/edu5b_eval_oldnll_${tag}_${SLURM_JOB_ID}.log" 2>&1 || true
done
for S in 50000 55000 60000 65000 70000; do
  F=$SCRATCH/ckpts/word_edu5b/ckpt_s$(printf '%07d' "$S").pt
  [ -f "$F" ] || continue
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/diag_v2.py "$F" "$AE" \
    --units-pkl "$OLDPKL" > "$LOGDIR/edu5b_mile_${S}_oldnll.log" 2>&1 || true
done
echo "[done] accept+diag $(date); logs:"; ls -la checkpoints/word_*_accept.log 2>/dev/null || true
