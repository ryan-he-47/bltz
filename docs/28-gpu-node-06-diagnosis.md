# 28 — gpu-v100s-06 诊断报告:驱动层整机卡死,不是坏卡(2026-09-15)

> 起因:临时会话扫集群 GPU 资源 → 复核 `AGENTS.md` 里那句"gpu-v100s-06(该节点坏)"。
> **结论:该节点 NVIDIA 驱动层整机卡死**——不是坏卡、不是硬件死。`slurmd` 正常响应,
> 所以 Slurm 报 `MIXED`、持续接活,而每个落到它上面的 job 都会永久挂死。
> **只有重启能解。**
> **拍板(用户 2026-09-15):本报告入档;`slurm/*.sbatch` 的
> `--exclude=gpu-v100s-06` 保留不动。** 报障文本单独落项目目录外(含他人账号标识,
> 不入即将开源的仓库)。

## 1. 结论与三条判据

一眼判死的三条(全部只读、秒级可得):

| 判据 | 健康 | gpu-v100s-06 |
|---|---|---|
| 读 `/proc/driver/nvidia/gpus/*/information` | 6 个条目全秒回 | **6 个条目全挂死** |
| D 状态(不可中断睡眠)进程数 | 0 | **117** |
| `CPULoad ÷ CPUAlloc` | ≤ 0.96 | **120.12 ÷ 8 = 15.0** |

**第一条是关键**:如果是坏卡,只会挂 1-2 个 PCI 条目;6 个全挂 = 驱动对整机 GPU 的
枚举路径整体堵死。

## 2. 证据链(同探针,06 vs 07 对照)

| 检查项 | gpu-v100s-07(健康) | gpu-v100s-06 |
|---|---|---|
| `/proc/.../gpus/*/information` | 6/6 秒回 | **6/6 挂死** |
| `nvidia-smi -L` | 正常 | 挂死(5 min 无输出) |
| 逐卡 CUDA init + matmul | 2/2 卡 `CUDA OK` | **4/4 卡挂死**(CUDA 序号 0–3) |
| D 状态进程 | 0 | 117 |
| `loadavg` | 4.78 | 121.13 |
| `CPULoad / CPUAlloc` | 4.43 / 21 = 0.21 | 120.12 / 8 = **15.0** |

驱动/内核:`NVRM 580.167.08`(2026-06-03 构建),kernel `5.14.0-687.41.1.el9_8.x86_64`,
Slurm 23.11.10,BootTime 2026-08-24,SlurmdStartTime 2026-08-27。

映射关系(踩坑记录):**Slurm 的 `IDX` 与 `CUDA 序号` 不是一回事**。06 上
`SLURM_JOB_GPUS=2,3,4,5` 对应 `CUDA_VISIBLE_DEVICES=0,1,2,3`;07 上
`SLURM_JOB_GPUS=4,5` 对应 `CUDA_VISIBLE_DEVICES=0,1`。`/proc/driver/nvidia/gpus/<pci>/information`
里的 `Device Minor` 字段才是 CUDA 序号,可用于反推映射。

## 3. 探针设计与两个致命坑(工程纪律,值得记住)

脚本:`slurm/probe_gpu_health.sbatch`(**本地产物,未入库**;集群副本在
`/gpfs1/home/yihe47/bltz/probe_gpu_health.sbatch`)。用法:

```bash
sbatch --nodelist=gpu-v100s-07 slurm/probe_gpu_health.sbatch                    # 健康对照
sbatch --nodelist=gpu-v100s-06 --gres=gpu:4 slurm/probe_gpu_health.sbatch       # 坏节点诊断
```

**坑 1:`timeout` 杀不掉 D 状态进程。** SIGTERM/SIGKILL 都无法中断不可中断睡眠,
所以"用 `timeout` 兜住可能挂死的调用"是**无效设计**——被堵住的 shell 永远走不到下一
步,整段探针被一张坏卡堵死。本会话为此连栽两次(`timeout 240` 包 CUDA init、
`timeout 10` 包 `/proc` 读),每次都把 job 耗到时限。
**正确做法:所有可能碰驱动/挂死的调用一律后台化写文件,主控 shell 只做
`sleep <固定窗口>` 然后回收结果**;谁没吐结果 = 谁挂死,用 `MARK_*` 标记定位它挂在哪一步。

**坑 2:`nvidia-smi -i <idx>` 在 580.167.08 已废弃**(报 `Option -i is not valid`),
逐卡 smi 是个无效测试,改整机 `nvidia-smi -L` 一次即可。

另:`ConstrainDevices=yes`,Slurm 按 job 白名单限制 `/dev`——探针**不能也不会**碰到
别人正在用的卡,这是能安全诊断的前提。

## 4. 因果链与时间线

```
2026-08-24   节点重启(NVRM 580.167.08)
2026-09-03   最早的用户僵尸出现(etime ≈ 12 天)
2026-09-04   ★ 我们的 MoB job 在 06 上连续挂死并 CANCELLED:
               mob_p4_joint / mob_p4_polish / mob_p4_fullft ×2 / mob_p5_twin
             其中两个进程至今仍是 D 状态僵尸(python / nvidia-smi,已卡 10 天+)
2026-09-04   gpu-v100s-05 被 TRIX-DRAINER 自动 drain("Not responding")
             gpu-v100s-06 ★ 逃过一劫 ★ —— slurmd 还活着,Slurm 只看这一项
2026-09-04→15  僵尸自催化堆积:新 job 挂死 → 又留新僵尸 → 117 个 → load 120
2026-09-15   本会话诊断结案
```

**`AGENTS.md` 那条"该节点坏"的真实来源就是这次 MoB 时代事故**,当时的绕过手段
(`--exclude=gpu-v100s-06`)被 bltz 继承下来,但原因没继承——所以才会出现"看起来已修好"
的误判空间。

**⚠️ 重要教训:挂死的 job 与健康长跑在 `sacct`/`squeue` 里完全一样。**
本次三个反面样本:
- 某账号 job 报 `RUNNING` 13h22m → 其实是卡死的 `python`(D 状态 13h22m);
- 另一账号 job 报 `RUNNING` 6h52m → 其实是卡死的 `nvidia-smi`(D 状态 7h02m);
- 一个跑满 5 天 `TIMEOUT` 的 job → 是挂死 5 天撞时限,不是健康长跑。

推论:**判断自己的 run 是否在推进,必须看 `train.log` 步号是否在涨,不能看 `squeue` 的状态。**

## 5. 全集群扫描:06 是唯一异常

判据 `CPULoad ÷ CPUAlloc`(A=0 时 load 应≈0):

```
explorer-01..06    0.96 0.96 0.12 0.40 0.12 0.25
gpu-a40-01/02      0.13 0.16
gpu-a100-03..12    0.30 0.13 0.15 0.43 0.86 0.84 0.78 0.67 0.54 0.82
gpu-v100s-01..04   0.32 0.77 0.46 0.54
gpu-v100s-05       drained
gpu-v100s-06       ★ 120.12 / 8 = 15.0 ★   ← 全集群 110 张卡里唯一异常
gpu-v100s-07       0.21
```

**起飞前体检(1 行,离线只读,建议纳入 §3.2 流程)**:

```bash
sinfo -N -o '%N|%C|%O|%G|%t'    # C=A/I/O/T CPU; O=负载; 看 O 是否远大于 A
```

## 6. 对 bltz 的影响与新增纪律

- **当前 run 未中招**:`bltz_2b`(job 556282)在 `gpu-v100s-03`,本会话验证时已到
  **step 3200、0.81s/step、loss 2.81**,正常推进。
- 本会话在 06 上消耗的探针:556365 / 556389 / 556438 / 556492(全部只读,全部
  已 scancel,均未影响其它 job)。
- **新增纪律(建议)**:
  1. 起飞前跑一次 §5 的体检行,`O > A` 的节点直接绕开;
  2. 判"在不在跑"只看 `train.log` 步号,不看 `squeue` 状态;
  3. 长期 job 不要落在 06;`--exclude` 保留(用户拍板);
  4. 探测可疑节点时,**每次挂死都会给该节点留永久僵尸**,不应反复提交。

## 7. 副作用与后续

- 我们 4 个探针 job 在 06 上新增约 15 个 D 状态僵尸(4 × CUDA init + 6 × `/proc` reader
  + smi)。这些 `SIGKILL` 无效,只能等重启;相对该节点原有 117 个、load 120,增量约 12%。
- 该节点同时**静默坑过至少 9 个其它账号**(名单留档在项目外的报障文本里)。
- 已向用户提出报障建议:请运维**重启 06(并考虑 05)**、期间 drain 06、给 TRIX-DRAINER
  加 `CPULoad vs CPUAlloc` 判据、并回查 09-03/04 是否有集群级事件(05 同日出事)。

## 8. 拍板回写

- 本报告入档 `docs/28`;`AGENTS.md` 与 `docs/08` 同步更新真实原因。
- **`slurm/*.sbatch` 的 `--exclude=gpu-v100s-06` 不动**(用户 2026-09-15 拍板);
  注意 `gpu-v100s-05` 才是真正 `DOWN+DRAIN` 的节点,目前反而没在 exclude 名单里——
  是否补上由用户另行决定。
- 报障文本落项目目录外:`E:\Trash_things\hpc-fault-report-gpu-v100s-06-2026-09-15.md`
  (含他人账号/job id,不入仓库)。
