#!/bin/bash
# Same-set adaptation/forgetting meter for the CPT run (2026-10-20 用户问
# 分布偏差 vs 能力回退): 用当前 ckpt 在【旧 FineWeb 评估样本】上跑 diag_v2
# —— held-out NLL 直接对表 p2 的 -40.69(同集同协议,分布变量被控制)。
# stingy 分区, 单卡, ~15 min。
#   sbatch slurm/bltz_cpt_heldout.sbatch
#SBATCH --job-name=bltz_cpt_heldout
#SBATCH --partition=stingy
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/cpt_heldout_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python

cd "$REPO"
git pull --ff-only || echo "[heldout] git pull failed, using existing checkout"

$PY -u scripts/diag_v2.py \
  "$SCRATCH/ckpts/word_cpt_s_stable/ckpt_full.pt" \
  "$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --units-pkl "$SCRATCH/data/fineweb10b_v2_word/eval_sample.pkl"

echo "[done] held-out eval on OLD FineWeb sample finished"
