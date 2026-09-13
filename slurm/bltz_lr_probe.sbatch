#!/bin/bash
# bltz POC - LR probe branch (2026-09-13 用户拍板): resume the main run's ckpt
# at a constant higher LR (1e-3) for 1000 steps -> slope comparison tells whether
# the stable-phase plateau is LR-limited. Separate ckpt dir; main run untouched.
#   sbatch slurm/bltz_lr_probe.sbatch
#SBATCH --job-name=bltz_lrprobe
#SBATCH --partition=gpu_v100s
#SBATCH --exclude=gpu-v100s-06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/lrprobe_%j.out

set -euo pipefail

SCRATCH=/gpfs1/scratch/yihe47/bltz
REPO=/gpfs1/home/yihe47/bltz/repo
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
CKPT=$SCRATCH/ckpts/lr_probe_final_1e3

cd "$REPO"
git pull --ff-only || echo "[probe] git pull failed, using existing checkout"
mkdir -p "$CKPT"

# freeze the FINAL ckpt of the main run (os.replace rotation is atomic, cp safe)
if [ ! -f "$CKPT/frozen.pt" ]; then
  cp "$SCRATCH/ckpts/twin/ckpt_full.pt" "$CKPT/frozen.pt"
fi

# dynamic steps = frozen step + 1000 (2026-09-13 lesson: hardcoded steps=10000
# made the first probe a 24s no-op — it queued so long the main run had already
# passed 10k, and range(12000, 10000) is empty)
STEPS=$($PY -c "import torch; s = torch.load('$CKPT/frozen.pt', map_location='cpu', weights_only=False); print(int(s['step']) + 1001)")
echo "[probe] frozen step + 1001 = $STEPS"

$PY -u scripts/train_bltz.py configs/default.yaml \
  --set data.cache_dir="$SCRATCH/data/fineweb10b" \
  --set train.bf16=false --set train.batch=16 \
  --set train.ckpt_dir="$CKPT" \
  --set train.resume="$CKPT/frozen.pt" \
  --set train.lr=1.0e-3 --set train.warmup=1 --set train.decay_frac=0.0 \
  --set train.steps="$STEPS" --set train.log_every=50

echo "[done] probe complete; compare $CKPT/train.log slope vs main run"
