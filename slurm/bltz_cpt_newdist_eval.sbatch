#!/bin/bash
# CPT 终态 新分布 held-out 评测 (2026-10-26 用户任务):
#   用 cpt_cache_main (DCLM60+Edu40, 新分布) 采样 eval pkl, 对 4 个 ckpt 跑
#   diag_v2 (终态 96292 -> 退火里程碑 95000 -> 平台终点 91205 -> p2 基线),
#   与旧分布 NLL 对表 (p2 旧分布 -40.69, 同协议) 分离"分布内禀熵差"与"能力提升"。
#   dump 是 CPU-only 但按登录节点纪律一律进 job, 与评测同一个 sbatch。
#   stingy 分区, 单卡 V100, dump + 4x diag_v2, 预计 ~25 min。
#     sbatch slurm/bltz_cpt_newdist_eval.sbatch
#SBATCH --job-name=bltz_cpt_newdist
#SBATCH --partition=stingy
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cpt_newdist_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
DATA=$SCRATCH/data
AE=$SCRATCH/ckpts/ae_48_256_word_full/best.pt
PKL=$DATA/cpt_main_eval.pkl

echo "=== [newdist] job $SLURM_JOB_ID start $(date '+%F %T') on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
git pull --ff-only || echo "[newdist] git pull failed, using existing checkout"

# ---- 1. eval pkl from the NEW distribution (CPU only; skip if already built) ----
if [ -f "$PKL" ]; then
  echo "=== dump: $PKL already exists ($(stat -c %s "$PKL") bytes), skip ==="
else
  echo "=== dump eval pkl: $DATA/cpt_cache_main -> $PKL ==="
  $PY -u scripts/dump_eval_pkl.py "$DATA/cpt_cache_main" "$PKL"
fi
stat -c '[newdist] pkl %n %s bytes' "$PKL"

# ---- 2. diag_v2 on 4 ckpts against the SAME pkl -----------------------------
run_diag () {
  tag="$1"; ckpt="$2"
  echo "=== ckpt $tag ==="
  if [ ! -f "$ckpt" ]; then
    echo "[skip] ckpt missing: $ckpt"
    return 0
  fi
  if ! $PY -u scripts/diag_v2.py "$ckpt" "$AE" --units-pkl "$PKL"; then
    echo "[fail] diag_v2 non-zero for $tag ($ckpt); recorded, continuing"
  fi
}

run_diag final_96292     "$SCRATCH/ckpts/word_cpt_s_anneal/last.pt"
run_diag milestone_95000 "$SCRATCH/ckpts/word_cpt_s_anneal/ckpt_s0095000.pt"
run_diag stable_91205    "$SCRATCH/ckpts/word_cpt_s_stable/last.pt"
run_diag p2_baseline     "$SCRATCH/ckpts/word_05b_p2/last.pt"

echo "=== [newdist] all ckpts done $(date '+%F %T') ==="
