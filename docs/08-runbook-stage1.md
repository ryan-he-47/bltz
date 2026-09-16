# 08 — Stage 1 runbook:缓存 + 训练(2026-09-09 起;2026-09-13 POC 策略修订)

> 压缩上下文后的无损起跑指南。顺序:AGENTS.md 当前状态 → 本文 → `docs/07` §3 命令手册。
> 设计意图在 `docs/01`,长期路线在 `docs/09`。

## 1. 状态快览(2026-09-13,全部已决/已建)

- 缓存:字节 + token 各 14/14 全量建成(scratch,`CACHE_READY` 已落)。
- 训练:**bltz 单臂** job 544172(batch 16 / fp16 / LR 4e-4 / WSD 20k 步 ≈ 14.5h)。
- 测试 9 个全绿;集群拓扑/纪律/监控见 §3.2;OOM 两轮假死根因见 `docs/07` §4.10。

## 2. Step 1 — 全量缓存构建(~5h,detached 后台,可断点续跑)

```powershell
$env:PYTHONPATH="E:\Trash_things\char_lm"
$py="E:\MiniConda\envs\cose2\python.exe"
# 字节 patch 缓存(14 parquet ~30GB 下载 + 8-worker 分割 ~4h)
Start-Process $py -ArgumentList "-u scripts\build_cache.py configs\default.yaml" `
  -RedirectStandardOutput checkpoints\build_cache.log -WindowStyle Hidden
# 完成后:token 缓存(parquet 复用 hf 下载缓存,phase 1 秒过,~1h)
Start-Process $py -ArgumentList "-u scripts\build_token_cache.py configs\baseline.yaml" `
  -RedirectStandardOutput checkpoints\build_token_cache.log -WindowStyle Hidden
```

- 断点:按 parquet 文件;残缺 shard(无 meta.json)自动丢弃重建;重跑同一命令即续。
- 验收:14 个 shard-NNNNN + meta.json;用 ShardReader/TokenShardReader 抽查
  (n_sequences、首条序列内容);记录 unit 统计(avg 单元字节数、总 units/tokens)。

## 3. Step 2 — 训练(策略变更 2026-09-13,用户拍板)

**先 bltz 单臂迭代,严格 baseline 缓跑。** 算力有限,优先把 bltz LM 本身
跑明白;BPB 弱参考用 gpt2/qwen/llama 同量级现成模型;等 bltz LM 收敛得
差不多了再补严格 baseline(基线臂脚本与 token 缓存已就绪,随时可挂)。
(原 twin 双臂预算表保留在下文作历史参考;缓存已全量建成:字节 14/14 +
token 14/14,2026-09-13。)

| 选项 | 改动 | 时长(11-15s/步) | 备注 |
|---|---|---|---|
| A 本机全量 | 无 | 20k ≈ 61-83h(2.5-3.5 天) | 风扇连转 |
| B 本机砍步 | `--set train.steps=8000` | ~24-33h | 预算缩水 |
| C S=256 | `--set data.n_patches=256`(本臂)+ `--set data.seq_len=320`(基线) | 20k ≈ 30-42h | 上下文减半,§12 已授权 |
| **D 集群(推荐)** | 见 §3.2 | 估 ~1 天内 | batch 32,fp16 就绪,本机解放 |

### 3.1 本机路径(A/B/C)

```powershell
# bltz 臂(S=512, batch 8, 20k 步)
Start-Process $py -ArgumentList "-u scripts\train_bltz.py configs\default.yaml" `
  -RedirectStandardOutput checkpoints\twin.out.log -WindowStyle Hidden
# 基线臂(T=640, batch 8, 20k 步)
Start-Process $py -ArgumentList "-u scripts\train_token_baseline.py configs\baseline.yaml" `
  -RedirectStandardOutput checkpoints\baseline_twin.out.log -WindowStyle Hidden
```

- **监控**:`Get-Content checkpoints\<run>\train.log -Tail 20 -Wait`(jsonl)。
  预期(**从头预训练**语境):长而缓的带噪下降,看千步级趋势,别看逐步
  抖动;不要套 MoB 蒸馏经验的"平台→decay 冲刺"形态(任务性质不同,
  2026-09-13 用户指正)。POC 阶段看趋势+定性分析,快速迭代,不做对照门。
- ckpt:`best.pt` / `last.pt` / `ckpt_full.pt`(.1 rotation)每 250 步(~50min);
  续训 `--set train.resume=checkpoints\<run>\ckpt_full.pt`。
- **优雅中断(腾出 GPU)**:往 run 目录放一个 STOP 文件即可——
  `New-Item checkpoints\<run>\STOP -ItemType File`;当前步结束后自动保存
  ckpt_full.pt 并干净退出(STOP 被消费,不会误伤后续续训)。前台进程也可
  Ctrl+C(第二次 Ctrl+C = 硬退,仍会尽力保存)。恢复:上面的 resume 命令。
- 双臂可同时开(各自 ~4GB,合计 ~8GB 临界——**建议串行**,或第二臂等第一臂
  过了 warmup 再开,防分配器抖动)。

### 3.2 集群路径(D,已落地 2026-09-12)

- **拓扑**:代码在 home `/gpfs1/home/yihe47/bltz/repo`(job 内 git clone 自
  GitHub,tarball fallback 在 `/gpfs1/scratch/yihe47/bltz_bootstrap.tar.gz`);
  数据/缓存/中间 ckpt 全在 scratch `/gpfs1/scratch/yihe47/bltz/`;
  日志 `~/bltz/logs/`,final+best 归档 `~/bltz/ckpt_final/`。
- **脚本**(`slurm/`,随仓库同步):bltz_cache.sbatch(CPU,**tiny 分区**——
  batch 分区拒绝 ≤10CPU/48G 的小 job)→ bltz_bench.sbatch(gpu_v100s,V100
  标定,`--dependency=afterok:<缓存job>`)→ bltz_train_twin.sbatch /
  bltz_train_baseline.sbatch(均:`--exclude=gpu-v100s-06`——该节点**驱动层整机卡死**,
  2026-09-15 确诊、未解,见 `docs/28`;fp16
  `--set train.bf16=false`,batch 16(bench 544090 实测:32 CUDA-OOM,16 =
  2602ms/step@23.2GB),5 天墙,`--signal=B:SIGTERM@60`
  时限/scancel 触发优雅保存,重提交自动 resume ckpt_full.pt)。
- **job 史**:缓存 543807/543893/543972 三轮 OOM 假死(根因 docs/07 §4.10)
  → 543983 建成;bench 544084(ShardReader OOM 三连杀,31ac21f 修复)
  → 544090 通过;**正式训练 544172**(20k 完成);LR 探针 544364(空跑事故)
  /546477(涨回,docs/10 §2);清理 546991;**60k 长 run 547403**
  (gcos 修正测量判良性后,docs/10 §3.2b)。
- **起飞前体检**(秒级只读,见 `docs/28` §5):`sinfo -N -o '%N|%C|%O|%G|%t'`——
  看 `O`(负载)是否远大于 `A`(已分配 CPU);健康节点比值 ≤1,`gpu-v100s-06` 是 15×。
- **监控**:`ssh -p 22 yihe47@burgundy.hpc.cityu.edu.hk "tail -20
  /gpfs1/home/yihe47/bltz/logs/<name>_<jobid>.out"`;缓存完成标记
  `/gpfs1/scratch/yihe47/bltz/CACHE_READY`。**判 run 在不在推进只看日志步号,不看
  `squeue` 状态**——挂死的 job 与健康长跑在 `sacct` 里完全一样(`docs/28` §4)。
- **注意**:HF 下载走 `HF_HOME=$SCRATCH/hf_cache`(防 home 配额);
  集群 cose2 = torch 2.5.1+cu121(fp16 路径已适配,numpy 2.4.6);
  `mob_race`(gpu-v100s-04)是用户其它在跑 job,**不许动**。
- 纪律:登录节点只跑秒级只读命令;**VPN 掉线 = 停手待命,不探测重试**。

## 4. Step 3 — POC 诊断(看趋势+定性,不设对照门)

1. **POC 诊断套件**:`scripts/diag_poc.py`(生成样本+涌现分段 / 梯度任务
   冲突 / H(Δ) 按词长与字符类分层 / 编码器几何探针 / 边界上下文分布 /
   Δ 与字节嵌入几何;全部定性描述,不设判据门槛)。**旧 D-1 判据已废止**
   (2026-09-13 用户拍板:武断、不贴合模型特性;docs/05 留档,diag_delta.py
   仅作原始曲线工具)。
2. **趋势判断**:train.log 千步级 loss 形态(从头预训练 = 长缓带噪下降,
   看趋势不看逐步抖动;勿套 MoB 蒸馏经验,2026-09-13 用户指正)。
3. **弱参考 sanity**:同级现成模型(gpt2/qwen/llama 级)做 BPB 锚点;
   **严格 BPB 对照 + FLOPs/byte 核算(D-5 协议)缓办**,随严格 baseline
   一起补(docs/01 D10)。
4. **记录**:趋势/定性结论入实验日志(体例继承 simple_point_cloud
   EXPERIMENT_LOG:数字、曲线、失败与修复,诚实边界)。

## 5. 失败预案

- **中断腾出算力(本机常态)**:`New-Item checkpoints\<run>\STOP -ItemType File`
  → 即存即退;事后 `--set train.resume=checkpoints\<run>\ckpt_full.pt` 接着跑。
  硬杀/断电最坏丢 250 步(~50min);SIGINT/SIGTERM 同样走保存路径。
- **spike**:guard(max(10×median, 2000))自动跳过,日志里 `event=spike_skip`;
  真崩溃 → kill → `--set train.resume=.../ckpt_full.pt` 续训;
  最新态已污染则用 `ckpt_full.pt.1`。降 LR 重启属新 run 决策,**问用户**。
- **OOM**:batch 8 已标定;仍 OOM → batch 6/4 或 `--set model.grad_ckpt=true`。
- **缓存损坏**:删坏 shard 目录重跑构建器(文件级断点)。
- **集群**:VPN 掉线=停手待命;坏节点已排除(`gpu-v100s-06` **驱动层整机卡死**、
  2026-09-15 确诊未解,见 `docs/28`;`gpu-v100s-05` 自 2026-09-04 起 DOWN+DRAIN);
  登录节点不跑重活。

## 6. 收尾

写 `docs/10-stage1-report.md` → commit → 向用户汇报 twin 对照结论 →
进 Stage 3(熵分割臂,规格见 `docs/09` §2;先打 git tag 冻结 Stage 1 配置)。
