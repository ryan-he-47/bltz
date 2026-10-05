#!/bin/bash
# word-arm 0.5B scaling, PHASE 2/2 (2026-09-30 拍板; 2026-10-05 p1 验货完成后
# 用户拍板发射): 平台期接续段, 至绝对步 50863(累计 10B units). 从 phase-1
# 平台期 pre-decay 存档点(ckpt_pre_decay.pt, step 20003 = 退火起点-1,
# weight-only)续训: 常量段 20000->45430, 退火尾 5433 步与 phase-1 同型
# (decay_frac 0.1068), final_frac=0. rewarmup 500(weight-only 重启 Adam
# 防护; peak 2e-4 = 平台 LR, 家规满足).
# 2026-10-05 发射前更新(用户拍板 A): ① resume 指针 -> ckpt_pre_decay.pt
# (RESUME 块: 本段任何重投自动改从 own ckpt_full 接续); ② rotate.resident
# 4->2 + mem 64G(p1 实测: 单 shard 4.85GB, 双 rank 4+1 顶穿 64G cgroup;
# 2+1=14.6GB/rank 安全); ③ 显式 pin attn_doc_mask=xformers.
# 墙外自动续投(~92.6h > 3 天墙): 墙前 300s 写 STOP(即存即退, 验证过的可靠
# 通道)-> 提前优雅退出; wait 后判段"未完成 且 近墙"-> sbatch 自投下一段。
# 手动 STOP 暂停不会误触续投(非近墙); 崩溃(非零退出)不自动重投。
#   sbatch slurm/bltz_word_05b_p2.sbatch
#SBATCH --job-name=bltz_word_05b_p2
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/word_05b_p2_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
TOK=/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B
CKPT=$SCRATCH/ckpts/word_05b_p2
WALL_S=$((3 * 86400))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

cd "$REPO"
git pull --ff-only || echo "[train] git pull failed, using existing checkout"
mkdir -p "$CKPT"

RESUME=(--set "train.resume=$SCRATCH/ckpts/word_05b_p1/ckpt_pre_decay.pt")
if [ -f "$CKPT/ckpt_full.pt" ]; then
  RESUME=(--set "train.resume=$CKPT/ckpt_full.pt")
  echo "[train] resuming own p2 state from $CKPT/ckpt_full.pt"
fi

# wall-out 自动续投: 墙前 300s 触发优雅停(后台看门狗, 与训练并行)
( sleep $((WALL_S - 300)); touch "$CKPT/STOP" ) &
WD=$!

$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29532 \
  scripts/train_bltz_v2.py configs/v2_word.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b_v2_word" \
  --set data.n_patches=4096 \
  --set data.rotate_shards.resident=2 --set data.rotate_shards.every=2000 \
  --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
  --set model.grad_ckpt=true --set model.attn_doc_mask=xformers \
  --set train.ae_ckpt="$SCRATCH/ckpts/ae_48_256_word_full/best.pt" \
  --set train.verify.tok_dir="$TOK" \
  --set train.bf16=false \
  --set train.fp16_init_scale=1024.0 --set train.fp16_growth_interval=500 \
  --set train.rewarmup=500 \
  --set train.batch=24 --set train.lr=2.0e-4 --set train.warmup=2000 \
  --set train.steps=50863 --set train.decay_frac=0.1068 --set train.final_frac=0.0 \
  --set train.ckpt_every=250 --set train.ckpt_keep=3 \
  --set train.milestone_every=5000 \
  --set train.ckpt_dir="$CKPT" \
  ${RESUME[@]+"${RESUME[@]}"} &
MAIN=$!

EXIT_CODE=0
wait $MAIN || EXIT_CODE=$?
kill $WD 2>/dev/null || true

if [ "$EXIT_CODE" -eq 0 ] && [ ! -f "$CKPT/last.pt" ] && [ "$SECONDS" -ge $((WALL_S - 900)) ]; then
  echo "[wall] near-wall graceful stop (elapsed ${SECONDS}s) -> resubmit next segment"
  sbatch "$REPO/slurm/bltz_word_05b_p2.sbatch" || echo "[wall] resubmit FAILED - manual resubmit needed"
fi

echo "[done] word-0.5B phase-2 (to 10B units) finished; log at $CKPT/train.log"
