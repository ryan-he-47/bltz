# 08 — 明日 runbook:全量缓存 + twin 双臂(2026-09-09)

> 压缩上下文后的无损起跑指南。顺序:AGENTS.md 当前状态 → 本文 → `docs/07` §3 命令手册。
> 设计意图在 `docs/01`,长期路线在 `docs/09`。

## 1. 起跑前检查单

- [ ] 磁盘 ~80GB 空闲(30GB parquet + ~50GB 字节缓存 + ~10GB token 缓存 + ckpts)
- [ ] GPU 空闲(`nvidia-smi`)、git 工作树干净
- [ ] **8 个测试全绿**:test_segment / test_model / test_cache / test_token_lm /
      test_shard_train / test_cache_builder / test_baseline_pipeline / test_resume
- [ ] 用户三项拍板:预算选项(§3)、磁盘确认、缓存下载放行

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

## 3. Step 2 — twin 双臂训练(预算选项,用户拍板)

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

- **监控**:`Get-Content checkpoints\<run>\train.log -Tail 20 -Wait`(jsonl);
  预期:loss 快降后平台,decay 段再降(MoB 经验:决定性收益在 decay tail)。
- ckpt:`best.pt` / `last.pt` / `ckpt_full.pt`(.1 rotation)每 250 步(~50min);
  续训 `--set train.resume=checkpoints\<run>\ckpt_full.pt`。
- **优雅中断(腾出 GPU)**:往 run 目录放一个 STOP 文件即可——
  `New-Item checkpoints\<run>\STOP -ItemType File`;当前步结束后自动保存
  ckpt_full.pt 并干净退出(STOP 被消费,不会误伤后续续训)。前台进程也可
  Ctrl+C(第二次 Ctrl+C = 硬退,仍会尽力保存)。恢复:上面的 resume 命令。
- 双臂可同时开(各自 ~4GB,合计 ~8GB 临界——**建议串行**,或第二臂等第一臂
  过了 warmup 再开,防分配器抖动)。

### 3.2 集群路径(D,推荐)

- 代码:git push 或 scp 同步。**缓存直接在集群构建**(MoB_Head 已验证集群可下
  FineWeb-Edu):sbatch CPU job 跑两个 build 脚本,
  `--set data.cache_dir=/gpfs1/home/yihe47/bltz/data/...`。
- 训练 sbatch 模板硬要求:`#SBATCH --exclude=gpu-v100s-06`(坏节点);
  fp16:`--set train.bf16=false`;V100 32GB:`--set train.batch=32`。
- 纪律:登录节点只跑秒级只读命令;**VPN 掉线 = 停手待命,不探测重试**。

## 4. Step 3 — 诊断与对照

1. **D-1(硬指标,本臂)**:
   `& $py -u scripts\diag_delta.py checkpoints\twin\best.pt configs\default.yaml`
   合格:per-k acc 衰减 + 逐 Δ 熵极差 > 0.05(docs/05 判据)。
2. **BPB 对照(D-5 协议,`docs/01` §9)**:同一 held-out 序列(缓存尾部):
   bltz 逐字节 CE(上一 patch 末 h 的 patch 内偏移处)vs 基线 next-token
   CE × (tokens/bytes)。**必须附 FLOPs/byte 核算**(docs/07 §5:基线 lm_head
   +34%/token;本臂头查询 ~341M/patch vs 骨干 226M/patch)。评测脚本若未建,
   按 D-5 协议现写(v2 欠账,`docs/07` §6)。
3. **生成样本**:本臂 θ_stop ∈ {0.5, 1.0, 2.0, ∞} 扫描(infer.py)vs 基线样本。
4. **报告**:`docs/10-stage1-report.md`(体例继承 simple_point_cloud
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
- **集群**:VPN 掉线=停手待命;坏节点已排除;登录节点不跑重活。

## 6. 收尾

写 `docs/10-stage1-report.md` → commit → 向用户汇报 twin 对照结论 →
进 Stage 3(熵分割臂,规格见 `docs/09` §2;先打 git tag 冻结 Stage 1 配置)。
