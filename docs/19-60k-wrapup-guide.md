# 19 — 60k 长 run 收尾任务指导(待用户通知后执行)

> **触发条件**:用户明确通知 60k 长 run(job **547403**,gpu_v100s)结束
> (跑完 / 优雅中断 / 异常)。本文即当时的执行手册,按序执行,不跳步。
> 项目已封存(docs/17),本任务是**最后的观测与归档**,不开启新研发。

## 0. 前置确认(秒级只读)

```powershell
ssh -p 22 yihe47@burgundy.hpc.cityu.edu.hk "squeue -j 547403 -o '%T %M %R' --noheader; tail -3 /gpfs1/scratch/yihe47/bltz/ckpts/twin60k/train.log"
```

- 若仍在跑且用户意图是中断:`New-Item` 无法 ssh 写(只读纪律)——改为
  通过 sbatch 一行 job 或 `scancel 547403`(SIGTERM 会触发优雅保存)。
  **中断前必须确认 `ckpt_full.pt` 的 mtime 已更新**。
- 异常形态(loss 爆炸 / gn 持续异常 / 中途崩):**先向用户报告,不擅自清理。**

## 1. 取数回本机

```powershell
$tmp = "C:\Users\he\AppData\Local\Temp\opencode"
scp -P 22 yihe47@burgundy.hpc.cityu.edu.hk:/gpfs1/scratch/yihe47/bltz/ckpts/twin60k/train.log "$tmp\train_60k.log"
scp -P 22 yihe47@burgundy.hpc.cityu.edu.hk:/gpfs1/home/yihe47/bltz/ckpt_final/60k/best.pt "$tmp\best_60k.pt"
scp -P 22 yihe47@burgundy.hpc.cityu.edu.hk:/gpfs1/home/yihe47/bltz/ckpt_final/60k/last.pt "$tmp\last_60k.pt"
# 里程碑清单(全段轨迹用):
ssh -p 22 yihe47@burgundy.hpc.cityu.edu.hk "ls /gpfs1/scratch/yihe47/bltz/ckpts/twin60k/ckpt_s*.pt"
```

## 2. 训练曲线分析(对照 20k)

```powershell
$env:PYTHONPATH="E:\Trash_things\char_lm"
& E:\MiniConda\envs\cose2\python.exe scripts\plot_train.py "$tmp\train_60k.log" viz\train_60k.png
```

读数要点(写进报告):稳定段斜率 vs 20k 的 -0.015/千步;decay 段(36k 起)
形态;gn 是否仍钉 0.1 还是随数据量抬起;skips 计数;**最终 loss 水平 vs
20k 的 2.66**;是否有二段下降或平台。

## 3. 终态 POC 诊断(diag_poc 全套,对照 docs/10)

```powershell
& E:\MiniConda\envs\cose2\python.exe scripts\diag_poc.py "$tmp\best_60k.pt"
# 分节慢跑可逐个:gen / gcos_ext / edelta / enc / bound / demb / bemb
```

对照口径:逐节与 `docs/10` §3 的 20k 数字并排(enc 语义信号是否出现、
edelta 词首-词尾结构是否加强、edelta 数字类 CE 4.5 是否下降、生成样本
质量与重复吸引子是否缓解、D11 词界停止下提交跨度分布)。全部定性描述,
不设门槛(D10)。

## 4. gcos 全段轨迹(本次有里程碑)

```powershell
# 拉回里程碑(12 份,weight-only 0.55GB/份;可按磁盘情况只拉 5 份代表:
# s0010000 / s0020000 / s0030000 / s0040000 / s0050000 / s0060000)
scp -P 22 "yihe47@burgundy.hpc.cityu.edu.hk:/gpfs1/scratch/yihe47/bltz/ckpts/twin60k/ckpt_s*.pt" "$tmp\traj\"
& E:\MiniConda\envs\cose2\python.exe scripts\diag_poc.py "$tmp\traj\ckpt_s0005000.pt" gcos_traj "$tmp\traj\ckpt_s0010000.pt" ...
```

读数(docs/11 §T5 规则):∇_h 余弦**向 0/正收敛 = 瞬态自愈**(印证 20k
结论);**持续走负 = 结构性**,如实记录(封存项目也不再干预,只留结论)。

## 5. 报告归档

- 新文档 `docs/20-60k-final-report.md`:曲线读数 + 诊断对照表 + 轨迹图与
  结论 + 与 20k 的全面对照;README 文档表与 AGENTS 状态回写;commit+push。

## 6. 集群清理(scratch 配额 300GB)

> **复工注记(2026-09-14 晚)**:本节为封存语境所写。**复工后缓存
> (`data/fineweb10b`、`data/tokengpt2`)是活资产,不删**;仅纯中间产物
> (rotation 旧档、探针目录、里程碑的冗余份)按配额酌情清理。

按 `slurm/bltz_cleanup.sbatch` 模板新起一次性 job(登录节点只读纪律,
删除一律走 job):
- 删 `data/fineweb10b`(63GB)与 `data/tokengpt2`(22GB)——缓存是可重建
  中间产物,封存后不再需要;
- 删 `ckpts/twin`(20k 中间态,保留 best/last 已在 home `ckpt_final/`)、
  `ckpts/lr_probe_*`(探针残留)、`ckpts/twin60k` 的 rotation 与
  里程碑(报告写完后;里程碑如需长期留存,只拷 2-3 份代表到 home);
- **保留**:`ckpt_final/`(best+last)、日志、bootstrap tarball。
- `hf_cache` 已在 546991 清理;`mob_race` 等用户其它资产**绝不动**。

清理完成后 `du -sh /gpfs1/scratch/yihe47/bltz` 记录最终占用,写进
docs/20 收尾段。项目正式闭账。
