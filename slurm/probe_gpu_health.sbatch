#!/bin/bash
# bltz GPU 节点健康探针(全非阻塞版, 2026-09-15)
#
# ============================ 设计要点(踩过的坑, 务必先读)============================
# 现象: gpu-v100s-06 上任何触碰该节点 GPU 驱动的调用都会**永久挂死**:
#   556365 卡在 `nvidia-smi -L`; 556389 卡在 `torch.cuda.is_available()`;
#   556438 卡在读 `/proc/driver/nvidia/gpus/0000:14:00.0/information`。
#   (对比: 该节点 CPULoad=119.48 而 CPUAlloc 仅 12; node07 对照 CPULoad=4.31)
#
# 【致命教训】**`timeout` 杀不掉 D 状态(uninterruptible sleep)进程**。
#   SIGTERM/SIGKILL 都无效, 所以"用 timeout 兜住挂死调用"是**无效设计** ——
#   被挂死调用堵住的 shell 永远走不到下一步, 整段探针被一张坏卡堵死。
#
# 因此本版:**任何可能碰驱动/挂死的调用一律后台化写入文件, 主控 shell 只做
#   `sleep <固定窗口>` 然后回收文件**。谁没吐结果 = 谁挂死, 用 MARK_* 标记定位
#   它挂在哪一步。这样一张坏卡不会阻塞其它卡的结论。
#
# 【副作用提醒】每次挂死都会在该节点留下无法回收的 D 状态进程(这正是它 CPULoad=119
#   的来源)。本探针仅用于确诊, 不应反复提交到同一坏节点。
#
# 安全边界: ConstrainDevices=yes, Slurm 按 job 白名单限制 /dev, 只能测本 job 分到的卡,
#          不会也不能干扰别人的卡; 不读写任何数据/ckpt。
#
# 用法(节点与卡数在命令行覆盖):
#   sbatch --nodelist=gpu-v100s-07 slurm/probe_gpu_health.sbatch            # 健康对照
#   sbatch --nodelist=gpu-v100s-06 --gres=gpu:4 slurm/probe_gpu_health.sbatch  # 坏节点建卡图
#
#SBATCH --job-name=gpu_probe
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/gpfs1/home/yihe47/bltz/logs/probe_%j.out

set -uo pipefail
PY=/gpfs1/home/yihe47/miniconda3/envs/cose2/bin/python
W=$(mktemp -d /tmp/probe.XXXXXX)
WINDOW=100

echo "=== bltz GPU 节点健康探针(全非阻塞版)==="
echo "node=$(hostname)  date=$(date -Is)  job=${SLURM_JOB_ID:-?}"
echo "Slurm IDX : ${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA 序号 : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "观察窗口  : ${WINDOW}s   (job 时限 5min)"
echo

echo "########## [0] 不碰驱动的即时信息 ##########"
echo -n "CPU 核数(nproc) : "; nproc
echo -n "/proc/loadavg   : "; timeout 5 cat /proc/loadavg 2>&1
echo -n "内存            : "; free -g 2>/dev/null | awk 'NR==2{print "total "$2"G  已用 "$3"G"}'
echo -n "/dev/nvidia*    : "; timeout 5 ls /dev/nvidia* 2>&1 | tr '\n' ' '; echo
echo
echo "D 状态(不可中断睡眠)进程 —— 负载虚高与'节点越用越烂'的直接来源:"
timeout 15 ps -eo stat,pid,user,etime,comm --no-headers 2>/dev/null | awk '$1 ~ /^D/' | head -n 30 | sed 's/^/    /'
echo -n "  D 状态计数     : "; timeout 15 ps -eo stat --no-headers 2>/dev/null | grep -c '^D'
echo -n "  本机可见进程数 : "; timeout 15 ps -e --no-headers 2>/dev/null | wc -l
echo

echo "########## [1] 后台启动所有可能挂死的调用 ##########"
PHYS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' ')
[ -z "$PHYS" ] && PHYS=$(timeout 5 ls /dev/nvidia[0-9]* 2>/dev/null | sed 's|.*/dev/nvidia||')
echo "本 job 分配到的 CUDA 序号: $PHYS"

( timeout 20 cat /proc/driver/nvidia/version > "$W/driver.txt" 2>&1 ) &
( timeout 20 ls /proc/driver/nvidia/gpus > "$W/gpulist.txt" 2>&1 ) &
# 整机一次 nvidia-smi -L。注意: 本驱动 580.167.08 已废弃 `-i`, 逐卡 smi 会直接报错
# (`Option -i is not valid for this command`), 那是个无效测试, 不要再按卡调用。
( echo "MARK_SMI_START"; timeout 20 nvidia-smi -L 2>&1; echo "MARK_SMI_DONE" ) > "$W/smi.txt" 2>&1 &

for P in $PHYS; do
  (
    echo "MARK_CUDA_START"
    timeout 45 env CUDA_VISIBLE_DEVICES="$P" "$PY" -u -c "
import torch
print('    is_available:', torch.cuda.is_available())
print('    name        :', torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device=0)
print('    matmul sum  :', float((x @ x).sum()))
torch.cuda.synchronize(0)
print('    >>> CUDA OK <<<')
" 2>&1
    echo "MARK_CUDA_DONE"
  ) > "$W/card_$P.txt" 2>&1 &
done

for d in /proc/driver/nvidia/gpus/*; do
  [ -e "$d" ] || continue
  b=$(basename "$d")
  ( timeout 15 cat "$d/information" > "$W/pci_$b.txt" 2>&1 ) &
done

echo "后台任务已启动, 进入 ${WINDOW}s 观察窗口(主控不会被任何坏卡阻塞)..."
sleep "$WINDOW"
echo "窗口结束, 回收结果:"
echo

echo "########## [2] 驱动层 ##########"
echo -n "/proc/driver/nvidia/version : "; cat "$W/driver.txt" 2>/dev/null | tr '\n' ' '; echo
echo    "/proc/driver/nvidia/gpus    : $(cat "$W/gpulist.txt" 2>/dev/null)"
for d in /proc/driver/nvidia/gpus/*; do
  [ -e "$d" ] || continue
  b=$(basename "$d")
  if [ -s "$W/pci_$b.txt" ]; then
    echo "[$b]"; sed 's/^/    /' "$W/pci_$b.txt"
  else
    echo "[$b]  >>> 无输出: 读驱动 proc 接口即挂死(驱动层就卡住, 不只是 nvidia-smi)<<<"
  fi
done
echo

echo "########## [3] 整机 nvidia-smi -L ##########"
if [ -s "$W/smi.txt" ]; then
  sed 's/^/  /' "$W/smi.txt"
  grep -q MARK_SMI_DONE "$W/smi.txt" || echo "  >>> 未跑完: nvidia-smi -L 挂死 <<<"
else
  echo "  >>> 无输出: nvidia-smi -L 挂死 <<<"
fi
echo

echo "########## [4] 逐卡结论(每卡独立进程, 一张坏卡不阻塞其它卡)##########"
for P in $PHYS; do
  echo "###### CUDA 序号 $P ######"
  f="$W/card_$P.txt"
  if [ ! -s "$f" ]; then
    echo "  >>> 无输出 <<<"
  else
    sed 's/^/  /' "$f"
    grep -q MARK_CUDA_DONE "$f" || echo "  >>> 卡死在 CUDA init/matmul 阶段 <<<"
    grep -q "CUDA OK" "$f"      && echo "  >>> 该卡可用 <<<"
  fi
  echo
done

rm -rf "$W"
echo "=== 探针结束 node=$(hostname) date=$(date -Is) ==="
