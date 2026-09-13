# 07 — 代码架构与运维手册(2026-09-09)

> 面向压缩上下文后的新会话:读完本文 + `docs/08-runbook-stage1.md` 即可无损起跑。
> 设计意图看 `docs/01-design.md`;本文件只讲代码现实。

## 1. 仓库地图

```
char_lm/
  AGENTS.md            项目约定 + 当前状态快照(先读)
  README.md            入口
  bltz/           主包
    config.py          Cfg: YAML + --set a.b=value(浮点覆盖必须带小数点)
    segment.py         固定语义分割器 + augment_unit_bytes(读时增强)
    data.py            build_batch(内存路径) + tensorize_units/collate_sequences + 流式取数
    shards.py          ShardWriter/ShardReader(字节 patch 单元流缓存)
    token_cache.py     TokenShardWriter/TokenShardReader(GPT-2 token 缓存,基线用)
    objectives.py      mtp_targets(查询/目标构建,唯一事实源) + mtp_loss
    trainer.py         train(): AdamW+WSD+bf16/fp16+spike guard,loss_fn 可注入(基线复用)
    infer.py           熵停止生成循环(朴素 O(T^2) 重算,KV cache 留 v2)
    models/
      encoder.py       Set Transformer(byte emb ⊕ 局部 PE → ISAB → PMA → latent)
      backbone.py      Llama 式骨干(RMSNorm/RoPE 半劈/SwiGLU,SDPA is_causal,grad_ckpt 选项)
      head.py          ConditionalByteHead (h, Δ) → 256 logits,Δ 恒为相对偏移
      model.py         BltzLM 接线
      token_lm.py      TokenLlama(GPT-2 词表 + 同一 Backbone + lm_head) + next_token_loss
  configs/
    default.yaml       bltz twin 权威配置(S=512, batch 8)
    baseline.yaml      token 基线 twin 配置(T=640, batch 8,含匹配数学)
    smoke.yaml         微型冒烟配置
  scripts/
    build_cache.py       字节缓存构建(两阶段并行,每 parquet 一 shard)
    build_token_cache.py token 缓存构建(同构,tokenizers Rust 批编码)
    train_token_baseline.py  基线训练入口
    smoke_overfit.py     冒烟:单 batch 过拟合(--set train.bf16=false 可验 fp16)
    smoke_tiny_train.py  冒烟:真实数据短跑 + θ_stop 扫描
    diag_delta.py        D-1 v2 诊断(逐 Δ / per-k / per-r 三表 + 判据)
    bench_twin.py        twin 规模步速/显存探针
    bench_parts.py       分模块计时
    prof_backbone.py     profiler:骨干前向
    prof_step.py         profiler:整步 fwd+bwd
  tests/               脚本式测试(逐个 python tests/test_xxx.py 直接跑,无 pytest)
  docs/                全部文档(见 README 表格)
  data/ checkpoints/ runs/ viz/   gitignored 产物
  reference_projects/  参考:BLT/Fast BLT 论文文本版、simple_point_cloud、MoB_Head
```

## 2. 关键数据结构

**训练 batch(dict of tensors,CPU 构建,trainer 内 .to(cuda))**:
`byte_ids` (B,S,L) long(PAD_ID=256)/ `pad_mask` (B,S,L) bool(True=pad)/
`flat` (B,S*L) long(拼接字节,>=flat_len 处为 0)/ `flat_len` (B,)/
`ends` (B,S) long(patch j 独占末尾偏移:patch j 占 flat[ends[j-1]:ends[j]])/
`patch_pos` (B,S*L) long(每个 flat 位置属于哪个 patch)。
构建:`data.build_batch`(内存)或 `ShardReader.make_batch`(缓存,可 augment)。

**MTP 查询(objectives.mtp_targets 唯一事实源)**:对每个 patch 位 j,
K_j = 未来 n 个 patch 的总长(序列尾截断);查询 Δ=1..K_j,目标 =
flat[ends[j]+Δ-1];权重 λ^(k-1),k = 目标落在第几个未来 patch;另返回
r(目标在 patch 内第几字节,D-1 分层用)。`mtp_loss` 分块算 CE(loss.query_chunk)。

**字节缓存 shard(shard-NNNNN/)**:`bytes.npy`(uint8 拼接)/
`unit_len.npy`(uint8)/ `unit_flag.npy`(bit0=文档首单元)/ `meta.json`。
**token 缓存 shard**:`tokens.npy`(uint16,文档间 EOS=50256)/ `doc_flag.npy`/ `meta.json`。
序列在读时切片,绝不跨 shard。断点:有 meta.json = 完成;无则删目录重建。

**ckpt(torch.save)**:`{model: state_dict, cfg: dict, step, loss}`;
train.ckpt_dir 下 `best.pt`(按 train loss)+ `last.pt`;诊断用
`Cfg(state["cfg"])` 重建模型,保证维度一致。
**断点 ckpt_full.pt**(+.1 rotation):全状态(model+opt+RNG+scaler+step+
med_hist/best/skips),每 `train.ckpt_every`(默认 250)步;**优雅中断**=
`<ckpt_dir>\STOP` 文件(detached 进程唯一通道)或 Ctrl+C/SIGTERM(第二次
Ctrl+C 硬退仍尽力存),循环顶检测、即存即退,STOP 用后自删;中断处最多
重复 1 步,绝不跳步。
**里程碑 ckpt_sNNNNNNN.pt**:每 `train.milestone_every`(默认 2000)步永久
保留(不轮替)——WSD 稳定段分支点,decay-on-demand / 探针 fork 用。
**weight-only**(0.55GB/份;scratch 配额 300GB 倒逼,2026-09-13):fork 分支
本来就重置优化器,精确断点恢复走 250 步 rotation 的 ckpt_full.pt(1.7GB
全状态)。从里程碑 resume 时日志打印 "weight-only resume",peak LR 不得超
母 run 结束 LR(house rule)。(事故记录:首轮 20k 中段快照因 rotation 只留
尾 2 而丢失,唯一幸存稳定段末快照在 scratch `lr_probe_1e3/frozen.pt`@~12000。)

**Cfg**:动态属性(load 时 setattr),LSP 报 "Cannot access attribute" 全是
误报;`--set a.b=value` 覆盖,**浮点必须带小数点**(`1.0e-8`,否则 yaml 当字符串)。

## 3. 命令手册

```powershell
$env:PYTHONPATH="E:\Trash_things\char_lm"
$py="E:\MiniConda\envs\cose2\python.exe"

# 测试(9 个,全绿才算起跑线就绪)
& $py tests\test_segment.py; & $py tests\test_model.py; & $py tests\test_cache.py
& $py tests\test_token_lm.py; & $py tests\test_shard_train.py
& $py tests\test_cache_builder.py; & $py tests\test_baseline_pipeline.py
& $py tests\test_resume.py; & $py tests\test_interrupt.py

# 全量缓存(明天第一步;~30GB 下载复用 hf cache,分割 ~4h / token ~1h)
& $py -u scripts\build_cache.py configs\default.yaml
& $py -u scripts\build_token_cache.py configs\baseline.yaml

# 训练(本机,前台短跑验证用;长任务一律 detached,见 docs/08 §监控)
& $py -u scripts\bench_twin.py --set data.cache_dir=data/fineweb10b
& $py -u scripts\train_token_baseline.py configs\baseline.yaml

# 诊断
& $py -u scripts\diag_delta.py checkpoints\<run>\best.pt configs\default.yaml

# 集群(V100,fp16):--set train.bf16=false;sbatch 模板必须 #SBATCH --exclude=gpu-v100s-06
```

## 4. 血泪坑位(全是实测,别再踩)

1. **RoPE 必须半劈连续 cat**(GPT-NeoX 式),禁止交错 stack(GPT-J 式)——
   后者在 WDDM 下每次 104ms、曾占骨干 CUDA 时间 67.7%(docs/06 §2)。
   **2026-09-08 晚换过约定**:旧 smoke ckpt(checkpoints/smoke/)与新代码不兼容,
   仅作历史 artifact,别拿它做新诊断。
2. **显存标定**(S=512, 138.5M):batch 32 OOM(~11GB)/ 16 分配器抖动(>30s/步)/
   **8 稳妥(~4GB)**。大 batch 先开 `model.grad_ckpt=true`。
3. **NumPy 2 的 uint8 标量参与 Python int 运算会溢出/报错**(shards.py 的
   pos 累加):循环里一律 `int(x)` 转换(shards.py 已有注释标记)。
4. `torch.from_numpy` 不能接只读 memmap(告警且行为未定义):一律 `.copy()`。
5. `torch.cuda.amp.GradScaler` 第一个位置参数是 init_scale **不是 device**;
   现用新 API `torch.amp.GradScaler("cuda", enabled=...)`(旧写法弃用告警,
   迁移见 commit 4500e0b)。
6. `expandable_segments:True` 在 Windows 上不支持(告警无害但无效)。
7. `Select-Object -Last N` 会缓冲到进程结束——长跑任务**不要**接在管道后面;
   用 Start-Process detached + 重定向到文件 + `Get-Content -Tail`。
8. HF datasets 流式在 Windows 退出时可能抛 WinError 10038(清理噪音,产物无恙)。
9. LSP 全部 import 报错(yaml/torch/regex/datasets/tokenizers/Cfg 属性)
   是解释器指向误报,环境在 cose2,以实际运行为准。
10. **整 shard 驻内存 + mp.Pool = OOM 假死**(2026-09-12 集群实测):
    ShardWriter/TokenShardWriter 曾用 Python list 缓冲全 shard(~8B/entry
    指针;token id >256 不被小整数缓存时 ~28B/entry),8 worker 峰值 45.6G
    顶 48G 限额,direct-reclaim 抖动 + OOM 杀 worker 后 **mp.Pool 永久挂死
    且无报错**(签名:日志停在仅一条 shard 完成打印、sstat AveCPU≈0.5 核、
    MaxRSS 贴限)。修复:array('b'/'H') 紧凑存储 + frombuffer 零拷贝
    (fec5058),每 worker ~1.2-2GB。教训:**构建器的全量内存必须实测**,
    本地小规模 dryrun 永远测不出全量内存剖面。

## 5. 性能事实(4060 本机实测,docs/06)

编码器大矩阵 ~5 TFLOPS(健康);骨干/头瘦矩阵 1-1.5 TFLOPS(就是这个速度);
twin 稳态 ≈ 11-15s/步(batch 8);**138.5M 参数**(SwiGLU 三矩阵 FFN 使骨干 113M,
超 124M 目标 12%,不影响 twin 对照——基线骨干同配方)。
**头查询是 FLOPs 大头**:每 patch 位置 ~15.5 个 Δ 查询 × 22M FLOPs ≈ 341M,
vs 骨干 226M/patch;BPB 对照报告必须附 FLOPs/byte 核算(baseline 侧 lm_head
+34%/token),不许只报参数量。

## 6. v2 工程欠账(按需开工,非明天必需)

- 推理 KV cache(infer.py 现为 O(T^2) 朴素重算)。
- 文档边界 block-causal(缓存里 unit_flag/doc_flag 已存好)。
- D-2 边界一致性脚本(diag_boundary.py);D-3 梯度余弦探针;
  D-4 patch 长度直方图;D-6 效率曲线;D-7 自验证一致率。
- BPB 评测脚本(D-5 协议:逐字节用上一 patch 末 h 的 patch 内偏移处 CE,
  与 token 基线公平,与 BLT 式字节 AR 不可直接比)。
