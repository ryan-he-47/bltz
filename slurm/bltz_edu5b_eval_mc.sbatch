#!/bin/bash
# edu5b 训后评估 · MC 电池(单卡,2026-10-30 GPU 紧张时拆分版)。
# 原 2 卡合并作业见 slurm/bltz_edu5b_eval.sbatch;因 gpu_v100s 只剩 1 张空闲卡,
# 拆成两个单卡作业:本文件跑 MC 8 项(主指标 acc_unit),bltz_edu5b_eval_accept.sbatch
# 跑内部四段 + diag NLL + 里程碑曲线。
# 三个 ckpt: p2pre(分支 45429)/ cpts7(CPT 同算力对照)/ edu5b(本实验 endpoint)。
#   sbatch slurm/bltz_edu5b_eval_mc.sbatch
#SBATCH --job-name=bltz_e5_mc
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/edu5b_eval_mc_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
VENV=$SCRATCH/venv_lmeval
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
LOGDIR=/gpfs1/home/yihe47/bltz/logs
OUTDIR=$SCRATCH/mc_eval
TASKS=arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,boolq,commonsense_qa

P2PRE=$SCRATCH/ckpts/word_05b_p2/ckpt_pre_decay.pt
CPTS7=$SCRATCH/ckpts/word_cpt_s_stable/ckpt_s0070000.pt
EDU5B=$SCRATCH/ckpts/word_edu5b/last.pt

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOGDIR" "$OUTDIR" checkpoints
[ -e checkpoints/protocol_table_500k.pt ] || \
  ln -s "$SCRATCH/protocol/protocol_table_500k.pt" checkpoints/protocol_table_500k.pt

for F in "$P2PRE" "$CPTS7" "$EDU5B"; do
  [ -f "$F" ] || { echo "[err] missing $F"; exit 2; }
done
[ -x "$VENV/bin/python" ] || { echo "[err] venv missing at $VENV"; exit 2; }

run_mc() {  # tag ckpt (GPU0 = the single allocated GPU)
  local tag=$1 ck=$2
  echo "[mc] start $tag ($ck) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u scripts/mc_eval.py \
    --ckpt "$ck" --ae "$AE" --tok "$TOK" --tag "$tag" \
    --tasks "$TASKS" --limit 0 --out-dir "$OUTDIR" \
    > "$LOGDIR/edu5b_eval_mc_${tag}_${SLURM_JOB_ID}.log" 2>&1 \
    || echo "[mc] $tag FAILED rc=$?"
  echo "[mc] done $tag $(date '+%T')"
}

echo "[job] host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | tr '\n' '|')"
run_mc edu5b "$EDU5B"
run_mc p2pre "$P2PRE"
run_mc cpts7 "$CPTS7"
echo "[done] MC battery $(date); results:"; ls -la "$OUTDIR"
