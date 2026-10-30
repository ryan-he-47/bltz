#!/bin/bash
# Post-training evaluation for the edu5b experiment (docs/50 §5.3).
#
# 2026-10-27 拍板: 评估预算延后到训练结束;E0(训练前 baseline)+E1(训练后 endpoint)
# 合并为**一个** 2-GPU 窗口,一次评三个 ckpt:
#   p2pre  = word_05b_p2/ckpt_pre_decay.pt       (分支 baseline, step 45429)
#   cpts7  = word_cpt_s_stable/ckpt_s0070000.pt   (同算力混合数据对照, +24571 步)
#   edu5b  = word_edu5b/last.pt                   (本实验 endpoint, +25432 步 = 5B units)
# 三者均须保留(prune 不执行; scripts/prune_ckpts.py 已把 s0070000 从删除清单剔除)。
#
# 2026-10-30 部署调整(训练完成后):
#   * 负载再平衡: GPU0 = mc(edu5b) + mc(p2pre);GPU1 = mc(cpts7) + run_accept ×3
#     + diag NLL + 里程碑曲线(旧版把 3 个 mc 全放 GPU0,GPU1 早退闲置)。
#   * mc_eval 的 PMI 默认 ON(argparse 只有 --no-pmi),旧脚本的 --pmi 是非法参数,已删。
#   * 旧词评测资产(集群原缺,2026-10-30 从本机上传,1.3GB + 209MB):
#       $SCRATCH/data/fineweb_word_local/            (旧 word 缓存;run_accept 的协议 pkl/cache)
#       $SCRATCH/protocol/protocol_table_500k.pt      (p2 验收同款 top-500k 协议表)
#     本脚本自动建 repo 侧软链: repo/data -> $SCRATCH/data,
#     repo/checkpoints/protocol_table_500k.pt -> $SCRATCH/protocol/...
#
#   sbatch slurm/bltz_edu5b_eval.sbatch
#SBATCH --job-name=bltz_edu5b_eval
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
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
mkdir -p "$LOGDIR" "$OUTDIR" checkpoints

# --- repo-side links for the old-word protocol assets (idempotent) ---
[ -e data ] || ln -s "$SCRATCH/data" data
[ -e checkpoints/protocol_table_500k.pt ] || \
  ln -s "$SCRATCH/protocol/protocol_table_500k.pt" checkpoints/protocol_table_500k.pt

for F in "$P2PRE" "$CPTS7" "$EDU5B"; do
  [ -f "$F" ] || { echo "[err] missing $F"; exit 2; }
done
for F in "$OLDPKL" "$OLDCACHE/eval_sample.pkl" checkpoints/protocol_table_500k.pt; do
  [ -e "$F" ] || { echo "[err] missing protocol asset $F"; exit 2; }
done
[ -x "$VENV/bin/python" ] || { echo "[err] venv missing at $VENV"; exit 2; }

# ---- CPU: edu held-out eval pkl (once; cheap, only reads ~300 sampled seqs) ----
if [ ! -f "$EDUPKL" ]; then
  echo "[prep] dumping $EDUPKL"
  $PY -u scripts/dump_eval_pkl.py "$EDUCACHE" "$EDUPKL" \
    > "$LOGDIR/edu5b_eval_dump_${SLURM_JOB_ID}.log" 2>&1
fi

run_mc() {  # args: tag ckpt device
  local tag=$1 ck=$2 dev=$3
  echo "[mc] start $tag dev=$dev ($ck) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$dev "$VENV/bin/python" -u scripts/mc_eval.py \
    --ckpt "$ck" --ae "$AE" --tok "$TOK" --tag "$tag" \
    --tasks "$TASKS" --limit 0 --out-dir "$OUTDIR" \
    > "$LOGDIR/edu5b_eval_mc_${tag}_${SLURM_JOB_ID}.log" 2>&1 \
    || echo "[mc] $tag FAILED rc=$?"
  echo "[mc] done $tag $(date '+%T')"
}

run_accept() {  # args: tag ckpt
  local tag=$1 ck=$2
  echo "[accept] start $tag ($ck) $(date '+%T')"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/run_accept.py "$ck" "$AE" --tag "$tag" \
    >> "$LOGDIR/edu5b_eval_accept_${SLURM_JOB_ID}.log" 2>&1 \
    || echo "[accept] $tag FAILED rc=$?"
  echo "[accept] done $tag $(date '+%T')"
}

(
  run_mc edu5b "$EDU5B" 0
  run_mc p2pre "$P2PRE" 0
) &
P0=$!

(
  run_mc cpts7 "$CPTS7" 1
  run_accept edu5b "$EDU5B"
  run_accept p2pre "$P2PRE"
  run_accept cpts7 "$CPTS7"
  # in-domain edu NLL (fit, not a capability metric) + old-dist NLL (forgetting)
  for pair in "p2pre:$P2PRE" "cpts7:$CPTS7" "edu5b:$EDU5B"; do
    tag=${pair%%:*}; ck=${pair#*:}
    CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/diag_v2.py "$ck" "$AE" --units-pkl "$EDUPKL" \
      > "$LOGDIR/edu5b_eval_edunll_${tag}_${SLURM_JOB_ID}.log" 2>&1 || true
    CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/diag_v2.py "$ck" "$AE" --units-pkl "$OLDPKL" \
      > "$LOGDIR/edu5b_eval_oldnll_${tag}_${SLURM_JOB_ID}.log" 2>&1 || true
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

echo "[job] launched: GPU0 mc(edu5b,p2pre) pid=$P0, GPU1 mc(cpts7)+accept×3+diag pid=$P1"
RC0=0; RC1=0
wait "$P0" || RC0=$?
wait "$P1" || RC1=$?
echo "[done] eval exit=$((RC0+RC1)) (gpu0=$RC0 gpu1=$RC1) elapsed=${SECONDS}s $(date)"
echo "[done] mc results:"; ls -la "$OUTDIR" 2>/dev/null || true
echo "[done] accept logs:"; ls -la checkpoints/word_*_accept.log 2>/dev/null || true
