#!/bin/bash
# 口径核对(2026-10-31,用户质疑 31%→27%):
#   2026-10-27 pilot 的 31.00% = CPT 终端 ckpt(ckpt_s0095000/last,word_cpt_s_last.pt)
#   + ARC-Easy **limit=300**(前 300 题)+ 本机 4060;
#   本次全量 = 三个不同 ckpt(p2pre/cpts7/edu5b)+ ARC-Easy **全量 2376 题** + 集群 V100。
# 本作业用当前代码在集群上重跑同一 pilot ckpt,对比 limit=300 与全量,验证差异是否来自
# "题目前缀子集"而非代码/口径变化。
#   sbatch slurm/bltz_edu5b_verify_arc.sbatch
#SBATCH --job-name=bltz_arc_verify
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/edu5b_arcverify_%j.out

set -euo pipefail
SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
VENV=$SCRATCH/venv_lmeval
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
OUT=$SCRATCH/mc_eval/arc_verify
LOG=$SCRATCH/ckpts/word_edu5b/arc_verify.log
CPTLAST=$SCRATCH/ckpts/word_cpt_s_anneal/last.pt
EDU5B=$SCRATCH/ckpts/word_edu5b/last.pt
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"

run() {  # tag ckpt limit
  local tag=$1 ck=$2 lim=$3
  echo "=== $tag ckpt=$ck limit=$lim $(date '+%T') ===" | tee -a "$LOG"
  "$VENV/bin/python" -u scripts/mc_eval.py --ckpt "$ck" --ae "$AE" --tok "$TOK" \
    --tag "$tag" --tasks arc_easy --limit "$lim" --out-dir "$OUT" 2>&1 | tee -a "$LOG"
}

cd "$REPO"
echo "[job] host=$(hostname) $(date)" | tee -a "$LOG"
run cptlast_300  "$CPTLAST" 300
run cptlast_full "$CPTLAST" 0
run edu5b_300    "$EDU5B"   300
echo "[done] $(date)" | tee -a "$LOG"
