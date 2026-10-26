#!/bin/bash
# 旧分布遗忘轨迹 + 退火缓存分布适应度, 4-ckpt 对表 (2026-10-26 用户任务):
#   old_word : p2 验收旧 eval pkl (本机 data/fineweb_word_local/eval_sample.pkl
#              已 scp 到 $SCRATCH/data/old_word_eval.pkl; 本机复现 p2 NLL -40.691 /
#              EM 0.2441 -> 跨机一致性校验锚)
#   anneal   : cpt_cache_anneal (web88.1/code5/math7, 15 shards/2.32B units) 现场 dump pkl
#   每张表 4 个 ckpt: stable 91205 -> milestone 95000 -> final 96292 -> p2 50863
#   stingy 单卡, dump + 8 x diag_v2, 预计 ~8 min。
#     sbatch slurm/bltz_cpt_milestone_eval.sbatch
#SBATCH --job-name=bltz_cpt_milestone
#SBATCH --partition=stingy
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cpt_milestone_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
DATA=$SCRATCH/data
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
OLD_PKL=$DATA/old_word_eval.pkl
ANNEAL_PKL=$DATA/cpt_anneal_eval.pkl

CK_STABLE=$SCRATCH/ckpts/word_cpt_s_stable/last.pt
CK_MILESTONE=$SCRATCH/ckpts/word_cpt_s_anneal/ckpt_s0095000.pt
CK_FINAL=$SCRATCH/ckpts/word_cpt_s_anneal/last.pt
CK_P2=$SCRATCH/ckpts/word_05b_p2/last.pt

echo "=== [milestone] job $SLURM_JOB_ID start $(date '+%F %T') on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
git pull --ff-only || echo "[milestone] git pull failed, using existing checkout"

# ---- 0. eval pkls: old = scp'd from workstation; anneal = dump here (cpu only) ----
if [ ! -f "$OLD_PKL" ]; then
  echo "[fatal] old-dist eval pkl missing: $OLD_PKL (should be scp'd from workstation)"
  exit 1
fi
stat -c '[milestone] old pkl %n %s bytes' "$OLD_PKL"

if [ -f "$ANNEAL_PKL" ]; then
  echo "=== dump: $ANNEAL_PKL already exists ($(stat -c %s "$ANNEAL_PKL") bytes), skip ==="
else
  echo "=== dump eval pkl: $DATA/cpt_cache_anneal -> $ANNEAL_PKL ==="
  $PY -u scripts/dump_eval_pkl.py "$DATA/cpt_cache_anneal" "$ANNEAL_PKL"
fi
stat -c '[milestone] anneal pkl %n %s bytes' "$ANNEAL_PKL"

# ---- 1. diag_v2 over the (pkl, ckpt) grid --------------------------------
run_diag () {
  pkl="$1"; ptag="$2"; tag="$3"; ckpt="$4"
  echo "=== $ptag $tag ==="
  if [ ! -f "$ckpt" ]; then
    echo "[skip] ckpt missing: $ckpt"
    return 0
  fi
  if ! $PY -u scripts/diag_v2.py "$ckpt" "$AE" --units-pkl "$pkl"; then
    echo "[fail] diag_v2 non-zero ($ptag $tag, $ckpt); recorded, continuing"
  fi
}

# old distribution (= p2 acceptance set): forgetting trajectory
run_diag "$OLD_PKL" old_word stable_91205    "$CK_STABLE"
run_diag "$OLD_PKL" old_word milestone_95000 "$CK_MILESTONE"
run_diag "$OLD_PKL" old_word final_96292     "$CK_FINAL"
run_diag "$OLD_PKL" old_word p2_baseline     "$CK_P2"

# annealed-data distribution (web88/code5/math7): adaptation check
run_diag "$ANNEAL_PKL" anneal stable_91205    "$CK_STABLE"
run_diag "$ANNEAL_PKL" anneal milestone_95000 "$CK_MILESTONE"
run_diag "$ANNEAL_PKL" anneal final_96292     "$CK_FINAL"
run_diag "$ANNEAL_PKL" anneal p2_baseline     "$CK_P2"

echo "=== [milestone] all ckpts done $(date '+%F %T') ==="
